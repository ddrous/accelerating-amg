from scipy.sparse import csr_matrix
from dgl.dataloading import GraphDataLoader

from model import to_prolongation_matrix_tensor, dgl_graph_to_sparse_matrices, AMGDataset
from dataset import DataSet
from multigrid_utils import P_square_sparsity_pattern

def model(A, coarse_nodes, baseline_P, C, graph_model, normalize_rows=True, normalize_rows_by_node=False):
    device = next(graph_model.parameters()).device

    # A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    pattern = P_square_sparsity_pattern(baseline_P, coarse_nodes)

    A_graph_dgl = AMGDataset(DataSet([A], [None], [coarse_nodes], [baseline_P], [pattern]))
    A_graph_dgl = A_graph_dgl.to(device)
    graph_dataloader = GraphDataLoader(A_graph_dgl, batch_size=1)

    P_graph_dgl = graph_model(next(iter(graph_dataloader)))

    [P_square_sparse], nodes = dgl_graph_to_sparse_matrices(P_graph_dgl, val_feature='P', return_nodes=True)

    # P_square_sparse = sparse_tensor_to_csr(P_square_sparse)
    P_dense, _ = to_prolongation_matrix_tensor(P_square_sparse, coarse_nodes, baseline_P, nodes,
                                                normalize_rows=normalize_rows,
                                                normalize_rows_by_node=normalize_rows_by_node)

    P_numpy = P_dense.cpu().detach().numpy()
    P_csr = csr_matrix(P_numpy)

    return P_csr


def baseline(A, splitting, baseline_P, C):
    return baseline_P