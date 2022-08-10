import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from scipy.sparse import csr_matrix
import os
import numpy as np


class AMGDataset(DGLDataset):
    """
    A class to convert an inhouse dataset (set of matrices) into a DGLdataset
    """
    def __init__(self, data, dtype=torch.float32, save_path=f"../data/unamed_data"):
        self.data = data
        self.dtype = dtype
        super(AMGDataset, self).__init__(name='AMG', save_dir=save_path)

    def process(self):
        As = self.data.As
        # Ss = self.data.Ss
        coarse_nodes_list = self.data.coarse_nodes_list
        # baseline_Ps = self.data.baseline_P_list
        sparsity_patterns = self.data.sparsity_patterns

        self.num_graphs = len(As)
        self.graphs = []
        dtype = self.dtype

        for i in range(self.num_graphs):
            ## Add edges features
            g = dgl.from_scipy(As[i], eweight_name='A')
            rows, cols = sparsity_patterns[i]
            A_coo = As[i].tocoo()

            # construct numpy structured arrays, where each element is a tuple (row,col), so that we can later use
            # the numpy set function in1d()
            baseline_P_indices = np.core.records.fromarrays([rows, cols], dtype='i,i')
            coo_indices = np.core.records.fromarrays([A_coo.row, A_coo.col], dtype='i,i')

            same_indices = np.in1d(coo_indices, baseline_P_indices, assume_unique=True)
            baseline_edges = same_indices.astype(np.float64)
            non_baseline_edges = (~same_indices).astype(np.float64)

            g = dgl.graph((A_coo.row, A_coo.col))

            g.edata['A'] = torch.as_tensor(A_coo.data, dtype=dtype).reshape((-1,1))
            g.edata['SP1'] = torch.as_tensor(baseline_edges, dtype=dtype).reshape((-1,1))
            g.edata['SP0'] = torch.as_tensor(non_baseline_edges, dtype=dtype).reshape((-1,1))
            # g.edata['P'] = torch.zeros_like(g.edata['A'])       ## <<----This will be the predicted P...... The values will be overwritten by the model

            ## Add node features
            coarse_indices = np.in1d(range(As[i].shape[0]), coarse_nodes_list[i], assume_unique=True)
            coarse_node_encodings = coarse_indices.astype(np.float64)
            fine_node_encodings = (~coarse_indices).astype(np.float64)

            g.ndata['C'] = torch.as_tensor(coarse_node_encodings, dtype=dtype).reshape((-1,1))
            g.ndata['F'] = torch.as_tensor(fine_node_encodings, dtype=dtype).reshape((-1,1))

            self.graphs.append(g)

        ## Delete data used for creation
        self.__dict__.pop('data', None)

    def to(self, device):
        """
        Move the dataset to GPU
        """
        for i in range(self.num_graphs):
            graph = self.graphs[i]
            self.graphs[i] = graph.to(device)
        return self

    def __getitem__(self, i):
        return self.graphs[i]

    def save(self):
        save_graphs(self.save_path, self.graphs)

    def load(self):
        self.graphs, _ = load_graphs(self.save_path)

    def __len__(self):
        return self.num_graphs

class AMGModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        h_feats = model_config.latent_size

        ## Encode nodes
        self.W1, self.W2, self.W3, self.W4 = self.create_MLP(2, h_feats, h_feats)

        ## Encode edges
        self.W5, self.W6, self.W7, self.W8 = self.create_MLP(3, h_feats, h_feats)

        ## Process
        self.conv1 = dglnn.SAGEConv(
                    in_feats=h_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
                    in_feats=2*h_feats, out_feats=h_feats, aggregator_type='mean')
        self.conv3 = dglnn.SAGEConv(
                    in_feats=2*h_feats, out_feats=h_feats, aggregator_type='mean')

        ## Decode edges
        self.W9, self.W10, self.W11, self.W12 = self.create_MLP(2*h_feats, h_feats, 1)    ## Concat source and dest before doing this

    def create_MLP(self, in_feats, hidden_feats, out_feats):
        W1 = nn.Linear(in_feats, hidden_feats)
        W2 = nn.Linear(hidden_feats, 4*hidden_feats)
        W3 = nn.Linear(4*hidden_feats, 2*hidden_feats)
        W4 = nn.Linear(2*hidden_feats, out_feats)
        return W1, W2, W3, W4

    def apply_MLP(self, h, W1, W2, W3, W4):
        return W4(F.relu(W3(F.relu(W2(F.relu(W1(h)))))))

    def encode_nodes(self, nodes):
        h = torch.cat([nodes.data['C'], nodes.data['F']], 1)
        return {'node_encs': self.apply_MLP(h, self.W1, self.W2, self.W3, self.W4)}

    def encode_edges(self, edges):
        h = torch.cat([edges.data['A'], edges.data['SP1'], edges.data['SP0']], 1)
        return {'edge_encs': self.apply_MLP(h, self.W5, self.W6, self.W7, self.W8)}

    def decode_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)          ##Key here
        # return {'P': self.W10(F.relu(self.W9(h))).squeeze(1)}
        return {'P': self.apply_MLP(h, self.W9, self.W10, self.W11, self.W12).abs().squeeze(1)}

    def forward(self, g):
        with g.local_scope():

            ## Encode nodes
            g.apply_nodes(self.encode_nodes)
            
            ## Encode edges
            g.apply_edges(self.encode_edges)

            ## Message passing
            n_encs = g.ndata['node_encs']
            e_encs = g.edata['edge_encs']

            h = self.conv1(g, n_encs, edge_weight=e_encs)
            h = F.relu(h)

            h = torch.cat([h, n_encs], 1)
            h = self.conv2(g, h, edge_weight=e_encs)
            h = F.relu(h)
            
            h = torch.cat([h, n_encs], 1)
            h = self.conv2(g, h, edge_weight=e_encs)

            ## Decode edges
            g.ndata['h'] = h
            g.apply_edges(self.decode_edges)


            P = g.edata['P']
            # return g.edata['P']
            # return g

        ### <<------- Trick to have local scope and keep newP ---------->>
        # if 'P' in g.edata:
        #     return g
        # else:
        g.edata['P'] = P
        return g


def dgl_graph_to_sparse_matrices(dgl_graph, val_feature='P', return_nodes=False):
    dgl_graph = dgl.unbatch(dgl_graph)
    num_graphs = len(dgl_graph)
    graphs = [dgl_graph[i] for i in range(num_graphs)]

    matrices = []
    nodes_lists = []
    for graph in graphs:
        indices = torch.stack(graph.edges(), axis=0)
        values = graph.edata[val_feature].squeeze()
        n_nodes = graph.num_nodes()
        matrix = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))
         # reordering is required because the pyAMG coarsening step does not preserve indices order
        matrix = matrix.coalesce()
        matrices.append(matrix)

    if return_nodes:
        for graph in graphs:
            nodes_list = graph.nodes()
            nodes_lists.append(nodes_list)
        return matrices, nodes_lists
    else: 
        return matrices


def to_prolongation_matrix_tensor(full_matrix, coarse_nodes, baseline_P, nodes,
                                  normalize_rows=False,
                                  normalize_rows_by_node=False):
    dtype = full_matrix.dtype
    device = full_matrix.device
    full_matrix = full_matrix.to_dense()

    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    num_rows = full_matrix.shape[0]
    new_diag = torch.ones(num_rows, device=device, dtype=dtype)
    full_matrix[range(num_rows), range(num_rows)] = new_diag
    
    # Select only columns corresponding to coarse nodes
    matrix = full_matrix[:, coarse_nodes]

    # Set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_P = torch.as_tensor(baseline_P.todense(), device=device, dtype=dtype)
    baseline_zero_mask = torch.as_tensor(torch.not_equal(baseline_P, torch.zeros_like(baseline_P)), 
                                            device=device, dtype=dtype)
    matrix = matrix * baseline_zero_mask

    raw_matrix = torch.clone(matrix)                      ## Unormalised tensor on which to apply the normalising loss

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = torch.as_tensor(nodes, device=device, dtype=dtype).reshape(-1,1)      ### Just nodes
        else:
            baseline_row_sum = torch.sum(baseline_P, dim=1, dtype=dtype).reshape(-1,1)         ### Basically just 1 

        # matrix_row_sum = torch.sum(matrix, dim=1, dtype=dtype).reshape(-1,1)
        matrix_row_sum = matrix.sum(dim=1).reshape(-1,1)

        # print("\nCOMPARE", baseline_row_sum)
        # print("\nCOMPARE TO", matrix_row_sum)

        # there might be a few rows that are all 0's - corresponding to fine points that are not connected to any
        # coarse point. We use "nan_to_num" to put these rows to 0's
        # matrix = torch.divide(matrix, torch.reshape(matrix_row_sum, (-1, 1)))

        matrix = matrix / matrix_row_sum
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # matrix = torch.nn.functional.normalize(matrix, p=1)

        matrix = matrix * baseline_row_sum

    ## Refill the square matrix with appropriate coarse columns and zeros for padding
    full_matrix[:, coarse_nodes] = matrix
    size = full_matrix.shape[0]
    non_coarse_nodes = np.setdiff1d(np.arange(size), coarse_nodes, assume_unique=1)
    zero_cols = torch.zeros((size, len(non_coarse_nodes)), device=device)
    full_matrix[:, non_coarse_nodes] = zero_cols

    return matrix, full_matrix, raw_matrix


def to_prolongation_matrix_csr(full_matrix, coarse_nodes, baseline_P, nodes, normalize_rows=True,
                               normalize_rows_by_node=False):
    """
    sparse version of the above function, for when the dense matrix is too large to fit in GPU memory
    used only for inference, so no need for backpropagation, inputs are csr matrices
    """

    # Use Scipy csr format
    inds = full_matrix.indices().cpu().detach().numpy()
    vals = full_matrix.values().cpu().detach().numpy()
    full_matrix = csr_matrix((vals, (inds[0], inds[1])))

    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    full_matrix.setdiag(np.ones(full_matrix.shape[0]))

    # # select only columns corresponding to coarse nodes
    matrix = full_matrix[:, coarse_nodes]
    # matrix = extract_coarse_cols_sparse(full_matrix, baseline_P.shape[0], coarse_nodes)       ## Implementation using Pytorch

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_P_mask = (baseline_P != 0).astype(np.float64)
    matrix = matrix.multiply(baseline_P_mask)
    matrix.eliminate_zeros()

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = baseline_P.sum(axis=1)
            baseline_row_sum = np.array(baseline_row_sum)[:, 0]

        matrix_row_sum = np.array(matrix.sum(axis=1))[:, 0]
        # https://stackoverflow.com/a/12238133
        matrix_copy = matrix.copy()
        matrix_copy.data /= matrix_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix_copy.data *= baseline_row_sum.repeat(np.diff(matrix_copy.indptr))
        matrix = matrix_copy

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    full_matrix[:, coarse_nodes] = matrix

    return matrix, full_matrix


def extract_coarse_cols_sparse(full_matrix, size, coarse_nodes):
    """
    A function to first of all add ones to the diagonal of a sparse matrix
    Then selects only the columns corresponding to the coarse nodes
    """
    device = full_matrix.device
    full_matrix  = full_matrix.coalesce()
    inds, vals = full_matrix.indices(), full_matrix.values()
    
    ## First, put ones on all diagonals
    diag = (inds[0]==inds[1])
    diag_inds, diag_vals = torch.arange(size).to(device), torch.ones(size).to(device)

    new_rows = torch.cat((inds[0][diag==False], diag_inds))
    new_cols = torch.cat((inds[1][diag==False], diag_inds))
    new_vals = torch.cat((vals[diag==False], diag_vals))

    ##, Now, filter only the coarse columns
    coarse_nodes.sort()
    coarse_size = len(coarse_nodes)
    coarse_nodes_dict = dict(zip(coarse_nodes, range(coarse_size)))
    coarse_mask = torch.isin(new_cols, torch.IntTensor(coarse_nodes).to(device))

    coarse_cols = new_cols[coarse_mask==True].to('cpu').apply_(coarse_nodes_dict.get).to(device)
    coarse_rows = new_rows[coarse_mask==True]
    coarse_vals = new_vals[coarse_mask==True]

    coarse_inds = torch.vstack((coarse_rows, coarse_cols))
    matrix_coo = torch.sparse_coo_tensor(coarse_inds, coarse_vals, size=(size, coarse_size)).coalesce()

    return matrix_coo.to_sparse_csr()


def get_model(model_name, model_config, train=False, train_config=None):
    checkpoint_dir = '../train_checkpoints/' + model_name
    if not os.path.isdir(checkpoint_dir):
        raise RuntimeError(f'training_dir {checkpoint_dir} does not exist')

    graph_model, optimizer, scheduler, global_step = load_model(checkpoint_dir, model_config,
                                                     train_config)

    if train:
        return graph_model, optimizer, scheduler, global_step
    else:
        graph_model.eval()      ## Eval mode
        return graph_model


def load_model(checkpoint_dir, model_config, train_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_dir + '/gnn_checkpoints.pth')

    model = AMGModel(model_config)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=50, min_lr=1e-6)
    # scheduler.load_state_dict(checkpoint['scheduler'])

    global_step = checkpoint['epoch']

    return model, optimizer, scheduler, global_step
