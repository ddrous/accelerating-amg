import copy
import os
import random
from sched import scheduler
import string

import fire
import matlab.engine
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import pyamg
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pyamg.classical.interpolate import direct_interpolation
from scipy.sparse import csr_matrix
from tqdm import tqdm

import configs
from data import generate_A
from dataset import DataSet
from model import AMGModel, dgl_graph_to_sparse_matrices, to_prolongation_matrix_tensor, AMGDataset
from multigrid_utils import block_diagonalize_A_single, block_diagonalize_P, two_grid_error_matrices, frob_norm, \
    two_grid_error_matrix, compute_coarse_A, P_square_sparsity_pattern, normalizing_loss, negative_loss, spectral_loss
from relaxation import relaxation_matrices
from utils import create_dir, create_results_dir, write_config_file, most_frequent_splitting, chunks, make_save_path


def create_dataset(num_As, data_config, run=0, matlab_engine=None):
    if data_config.load_data:
        As_filename = f"../data/periodic_delaunay_num_As_{num_As}_num_points_{data_config.num_unknowns}" \
            f"_rnb_{data_config.root_num_blocks}_epoch_{run}.npy"
        if not os.path.isfile(As_filename):
            raise RuntimeError(f"file {As_filename} not found")
        As = np.load(As_filename)

        # workaround for data generated with both matrices and point coordinates
        if len(As.shape) == 1:
            As = list(As)
        elif len(As.shape) == 2:
            As = list(As[0])
    else:
        As = [generate_A(data_config.num_unknowns,
                         data_config.dist,
                         data_config.block_periodic,
                         data_config.root_num_blocks,
                         add_diag=data_config.add_diag,
                         matlab_engine=matlab_engine) for _ in range(num_As)]

    if data_config.save_data:
        As_filename = f"../data/periodic_delaunay_num_As_{num_As}_num_points_{data_config.num_unknowns}" \
            f"_rnb_{data_config.root_num_blocks}_epoch_{run}.npy"
        np.save(As_filename, As)
    return create_dataset_from_As(As, data_config, matlab_engine)


def create_dataset_from_As(As, data_config, matlab_engine=None):
    if data_config.block_periodic:
        Ss = [None] * len(As)  # relaxation matrices are only created per block when calling loss()
    else:
        Ss = relaxation_matrices(As)
    if data_config.block_periodic:
        orig_solvers = [pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
                        for A in As]
        # for efficient Fourier analysis, we require that each block contains the same sparsity pattern - set of
        # coarse nodes, and interpolatory set for each node. The AMG C/F splitting algorithms do not output the same
        # splitting for each block, but the blocks are relatively similar to each other. Taking the most common set
        # of coarse nodes and repeating it for each block might be a good strategy
        splittings = []
        baseline_P_list = []
        for i in range(len(As)):

            orig_splitting = orig_solvers[i].levels[0].splitting
            block_splittings = list(chunks(orig_splitting, data_config.num_unknowns))
            common_block_splitting = most_frequent_splitting(block_splittings)
            if np.all(common_block_splitting==0):       ## Just make sure a few fine nodes are selected
                repeated_splitting = orig_splitting
            else:
                repeated_splitting = np.tile(common_block_splitting, data_config.root_num_blocks ** 2)
            splittings.append(repeated_splitting)

            # we recompute the Ruge-Stuben prolongation matrix with the modified splitting, and the original strength
            # matrix. We assume the strength matrix is block-circulant (because A is block-circulant)
            A = As[i]
            C = orig_solvers[i].levels[0].C     ## classical_strength_of_connection between nodes
            P = direct_interpolation(A, C, repeated_splitting)
            baseline_P_list.append(P)
            # baseline_P_list.append(torch.as_tensor(P.toarray(), dtype=torch.float64))

        coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]

    else:
        solvers = [pyamg.ruge_stuben_solver(A, max_levels=2, keep=True, CF=data_config.splitting)
                   for A in As]
        baseline_P_list = [solver.levels[0].P for solver in solvers]
        # baseline_P_list = [torch.as_tensor(P.toarray(), dtype=torch.float64) for P in baseline_P_list]
        splittings = [solver.levels[0].splitting for solver in solvers]
        coarse_nodes_list = [np.nonzero(splitting)[0] for splitting in splittings]

    spasity_patterns_list = []
    for A, coarse_nodes, baseline_P in zip(As, coarse_nodes_list, baseline_P_list):
        pattern = P_square_sparsity_pattern(baseline_P, coarse_nodes)
        spasity_patterns_list.append(pattern)

    return DataSet(As, Ss, coarse_nodes_list, baseline_P_list, spasity_patterns_list)


def loss(dataset, P_graphs_dgl, run_config, train_config, data_config):

    As = dataset.As
    Ps_square, nodes_list = dgl_graph_to_sparse_matrices(P_graphs_dgl, val_feature='P', return_nodes=True)

    if train_config.fourier:
        As = [torch.as_tensor(A.todense(), dtype=torch.complex64) for A in As]
        block_As = [block_diagonalize_A_single(A, data_config.root_num_blocks, tensor=True) for A in As]
        block_Ss = np.array(relaxation_matrices([csr_matrix(block) for block_A in block_As for block in block_A]))

    # batch_size = len(dataset.coarse_nodes_list)         ##<<<----- WHY ?????
    batch_size = len(Ps_square)         ##<<<----- WHY ?????
    total_norm = torch.tensor(0.0, requires_grad=True)
    for i in range(batch_size):
        if train_config.fourier:
            num_blocks = data_config.root_num_blocks ** 2 - 1

            P_square = Ps_square[i]
            coarse_nodes = dataset.coarse_nodes_list[i]
            baseline_P = dataset.baseline_P_list[i]
            nodes = nodes_list[i]
            P, full_P, _ = to_prolongation_matrix_tensor(P_square, coarse_nodes, baseline_P, nodes,
                                              normalize_rows=run_config.normalize_rows,
                                              normalize_rows_by_node=run_config.normalize_rows_by_node)
            block_P = block_diagonalize_P(full_P, data_config.root_num_blocks, coarse_nodes)

            As = torch.stack(block_As[i])
            Ps = torch.as_tensor(torch.stack(block_P), device=As.device, dtype=As.dtype)
            Rs = torch.conj(torch.transpose(Ps, dim0=1, dim1=2))
            Ss = torch.as_tensor(block_Ss[num_blocks * i:num_blocks * (i + 1)], device=As.device, dtype=As.dtype)

            # print("Number of Ps NaNs", torch.nonzero(torch.isnan(Rs)))

            Ms = two_grid_error_matrices(As, Ps, Rs, Ss)
            M = Ms[-1]  # for logging
            block_norms = torch.abs(frob_norm(Ms, power=1))

            block_max_norm = torch.max(block_norms)
            total_norm = total_norm + block_max_norm

        else:
            P_square = Ps_square[i]
            coarse_nodes = dataset.coarse_nodes_list[i]
            baseline_P = dataset.baseline_P_list[i]
            nodes = nodes_list[i]

            P, _, P_unnormed = to_prolongation_matrix_tensor(P_square, coarse_nodes, baseline_P, nodes,
                                              normalize_rows=True, ## Row-normalisation is enforced through a loss function too
                                              normalize_rows_by_node=False)

            R = torch.transpose(P, dim0=-2, dim1=-1)
            S = torch.as_tensor(dataset.Ss[i], dtype=P.dtype, device=P.device)
            A = torch.as_tensor(As[i].todense(), dtype=P.dtype, device=P.device)

            M = two_grid_error_matrix(A, P, R, S)

            ## A loss fucntion to minimize the frobenius norm
            # frob_loss = frob_norm(M)
            frob_loss = spectral_loss(M)

            ## A loss function to enforce the row-wize sum = 1
            # true_or_false = torch.as_tensor(run_config.normalize_rows, dtype=P.dtype)
            norm_loss, P_normed = normalizing_loss(P_unnormed)
            # norm_loss = norm_loss * true_or_false

            neg_loss = negative_loss(P_unnormed)

            eps = 0.05
            total_norm = total_norm + (eps)*frob_loss + (1)*norm_loss + (1)*neg_loss

    return total_norm / batch_size, M  # M is chosen randomly - the last in the batch


def save_model_and_optimizer(checkpoint_prefix, model, optimizer, scheduler, global_step):
    torch.save({
            'epoch': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_prefix, _use_new_zipfile_serialization=False)


def train_run(run_dataset, run, batch_size, config,
              model, optimizer, checkpoint_prefix,
              eval_dataset, eval_A_dgl, eval_config, 
              device="cpu", nb_iter_batch=[None], 
              tb_writer=None, scheduler=None):
    num_As = len(run_dataset.As)
    if num_As % batch_size != 0:
        raise RuntimeError("batch size must divide training data size")

    run_dataset = run_dataset.shuffle()
    num_batches = num_As // batch_size
    loop = tqdm(range(num_batches))
    for batch in loop:
        start_index = batch * batch_size
        end_index = start_index + batch_size
        batch_dataset = run_dataset[start_index:end_index]

        save_path = make_save_path(config.data_config.dist, len(batch_dataset.As), 
                                    config.data_config.num_unknowns,
                                    config.data_config.root_num_blocks)
        batch_A_dgl_dataset = AMGDataset(batch_dataset, dtype=torch.float32, save_path=save_path)
        batch_A_dgl_dataset_gpu = batch_A_dgl_dataset.to(device)        ## Move data to GPU if available

        batch_dataloader = GraphDataLoader(batch_A_dgl_dataset_gpu, batch_size=batch_size)       ## Only 1 batch can be made
        batch_P_dgl_dataset = model(next(iter(batch_dataloader)))
        # print("DOC:", batch_P_dgl_dataset.edata['P'])
        # print("Number of Graph NaNs", torch.nonzero(torch.isnan(batch_P_dgl_dataset.edata['P'].view(-1))))

        total_loss, M = loss(batch_dataset, batch_P_dgl_dataset,
                            config.run_config, config.train_config, config.data_config)

        print(f"total_loss: {total_loss.item()} \t \t \t current_learning_rate: {optimizer.param_groups[0]['lr']}")
        save_every = max(1000 // batch_size, 1)
        if batch % save_every == 0:
            save_model_and_optimizer(checkpoint_prefix, model, optimizer, scheduler, int(batch))      ## <------ Find a better way to get the global_step (the number count for the batches)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        nb_iter_batch[0] += 1

        variables = model.named_parameters()

        if tb_writer is not None:
            record_tb(M, run, num_As, batch, batch_size, total_loss.item(), loop, model,
                  variables, eval_dataset, eval_A_dgl, eval_config, nb_iter_batch[0], tb_writer)

        if scheduler:
            scheduler.step(total_loss)


def record_tb(M, run, num_As, batch, batch_size, frob_loss, loop, model,
              variables, eval_dataset, eval_A_dgl, eval_config, iter_nb, tb_writer):
    batch = run * num_As + batch

    record_loss_every = max(1 // batch_size, 1)
    if batch % record_loss_every == 0:
        record_tb_loss(frob_loss, iter_nb, tb_writer)

    record_params_every = max(300 // batch_size, 1)
    if batch % record_params_every == 0:
        record_tb_params(batch_size, loop, variables, iter_nb, tb_writer)

    record_spectral_every = max(300 // batch_size, 1)
    if batch % record_spectral_every == 0:
        record_tb_spectral_radius(M, model, eval_dataset, eval_A_dgl, eval_config, iter_nb, tb_writer)


def record_tb_loss(frob_loss, iter_nb, tb_writer):
    tb_writer.add_scalar("loss", frob_loss, iter_nb)


def record_tb_params(batch_size, loop, variables, iter_nb, tb_writer):

    avg_time = getattr(loop, "avg_time", None)
    if avg_time is not None:
        tb_writer.add_scalar('seconds_per_batch', torch.as_tensor(avg_time), iter_nb)

    for name, var in variables:
        variable = var.data
        grad = var.grad
        variable_name = name

        if grad is not None:
            # grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)      ## Remeber to Delete this !
            # variable = torch.nan_to_num(variable, nan=0.0, posinf=0.0, neginf=0.0)      ## Delete this !

            tb_writer.add_scalar(variable_name + '_grad', torch.norm(grad) / batch_size, iter_nb)
            # tb_writer.add_histogram(variable_name + '_grad_histogram', grad / batch_size, iter_nb)
            tb_writer.add_scalar(variable_name + '_grad_fraction_dead', torch.count_nonzero(grad)/torch.numel(grad), iter_nb)
            tb_writer.add_scalar(variable_name + '_value', torch.norm(variable), iter_nb)
            # tb_writer.add_histogram(variable_name + '_value_histogram', variable, iter_nb)


def record_tb_spectral_radius(M, model, eval_dataset, eval_A_dgl, eval_config, iter_nb, tb_writer):

    # spectral_radius = np.abs(np.linalg.eigvals(M.detach().numpy())).max()
    spectral_radius = torch.abs(torch.linalg.eigvals(M)).max()
    tb_writer.add_scalar('spectral_radius', spectral_radius, iter_nb)

    ###<<< ------ Send model to GPU before doing this one ------->> REVALUATE IT
    eval_dataloader = GraphDataLoader(eval_A_dgl, batch_size=len(eval_A_dgl))
    eval_P_dgl_dataset = model(next(iter(eval_dataloader)))

    eval_loss, eval_M = loss(eval_dataset, eval_P_dgl_dataset,
                                eval_config.run_config,
                                eval_config.train_config,
                                eval_config.data_config)

    # eval_spectral_radius = np.abs(np.linalg.eigvals(eval_M.detach().numpy())).max()
    eval_spectral_radius = torch.abs(torch.linalg.eigvals(eval_M)).max()
    tb_writer.add_scalar('eval_loss', eval_loss, iter_nb)
    tb_writer.add_scalar('eval_spectral_radius', eval_spectral_radius, iter_nb)


def coarsen_As(fine_dataset, model, batch_size=64):
    # computes the Galerkin operator P^(T)AP on each of the A matrices in a batch, using the Prolongation
    # outputted from the model
    As = fine_dataset.As
    device = next(model.parameters()).device
    coarse_nodes_list = fine_dataset.coarse_nodes_list
    baseline_P_list = fine_dataset.baseline_P_list

    batch_size = min(batch_size, len(As))
    num_batches = len(As) // batch_size

    batched_As = list(chunks(As, batch_size))
    batched_Ss = [None for A in batched_As]
    batched_coarse_nodes_list = list(chunks(coarse_nodes_list, batch_size))
    batched_baseline_P_list = list(chunks(baseline_P_list, batch_size))

    spasity_patterns_list = []
    for A, coarse_nodes, baseline_P in zip(As, coarse_nodes_list, baseline_P_list):
        pattern = P_square_sparsity_pattern(baseline_P, coarse_nodes)
        spasity_patterns_list.append(pattern)

    batched_spasity_patterns_list = list(chunks(spasity_patterns_list, batch_size))

    save_path = make_save_path("coarsened_batch_garlekin_As", len(batched_As), 4, 4)
    A_graphs_dgl_batches = [AMGDataset(DataSet(batch_As, batch_Ss, 
                                        batch_coarse_nodes_list, 
                                        batch_baseline_P_list, 
                                        batch_sp_list), dtype=torch.float32, save_path=save_path)
                              for batch_As, batch_Ss, batch_coarse_nodes_list, batch_baseline_P_list, batch_sp_list
                              in zip(batched_As, batched_Ss, batched_coarse_nodes_list, batched_baseline_P_list, batched_spasity_patterns_list)]

    Ps_square = []
    nodes_list = []
    for batch in tqdm(range(num_batches)):
        A_graphs_dgl = A_graphs_dgl_batches[batch].to(device)
        batch_dataloader = GraphDataLoader(A_graphs_dgl, batch_size=batch_size)
        P_graphs_dgl = model(next(iter(batch_dataloader)))
        P_square_batch, nodes_batch = dgl_graph_to_sparse_matrices(P_graphs_dgl, val_feature='P', return_nodes=True)
        Ps_square.extend(P_square_batch)
        nodes_list.extend(nodes_batch)

    coarse_As = []
    for i in tqdm(range(len(As))):
        P_square = Ps_square[i]
        nodes = nodes_list[i]
        coarse_nodes = coarse_nodes_list[i]
        baseline_P = baseline_P_list[i]
        P, _, _ = to_prolongation_matrix_tensor(P_square, coarse_nodes, baseline_P, nodes, normalize_rows=True)
        R = torch.transpose(P, dim0=-2, dim1=-1)
        A_csr = As[i]
        A = torch.as_tensor(A_csr.toarray(), dtype=torch.float32, device=device)
        tensor_coarse_A = compute_coarse_A(R, A, P)
        coarse_A = csr_matrix(tensor_coarse_A.cpu().detach().numpy())
        coarse_As.append(coarse_A)

    return coarse_As


def create_coarse_dataset(fine_dataset, model, data_config, run_config, matlab_engine):
    As = coarsen_As(fine_dataset, model)
    return create_dataset_from_As(As, data_config)


def train(config='GRAPH_LAPLACIAN_TRAIN', eval_config='FINITE_ELEMENT_TEST', seed=3):
    config = getattr(configs, config)
    # config = getattr(configs, 'GRAPH_LAPLACIAN_TRAIN')        ## Use this to avoid recreating the dataset all the time
    eval_config = getattr(configs, eval_config)
    # eval_config = getattr(configs, eval_config)
    # eval_config.run_config = config.run_config

    ##-------------->> ACCELERATION CHOICES <<-----------------
    matlab_engine = matlab.engine.start_matlab()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fix random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    matlab_engine.eval(f'rng({seed})')

    batch_size = min(config.train_config.samples_per_run, config.train_config.batch_size)

    # we measure the performance of the model over time on one larger instance that is not optimized for
    # eval_dataset = create_dataset(1, eval_config.data_config)
    eval_dataset = create_dataset(1280, config.data_config, run=0, matlab_engine=matlab_engine)

    save_path = make_save_path(eval_config.data_config.dist, len(eval_dataset.As), 
                                    eval_config.data_config.num_unknowns,
                                    eval_config.data_config.root_num_blocks)
    eval_dataset_dgl = AMGDataset(eval_dataset, dtype=torch.float32, save_path=save_path)
    eval_dataset_dgl = eval_dataset_dgl.to(device)

    if config.train_config.load_model:
        raise NotImplementedError()
    else:
        model = AMGModel(config.model_config)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train_config.learning_rate)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=20, min_lr=1e-6)

    run_name = ''.join(random.choices(string.digits, k=5))  # to make the run_name string unique
    # run_name = '00000'  # all runs have same name
    create_results_dir(run_name)
    write_config_file(run_name, config, seed)

    checkpoint_prefix = config.train_config.checkpoint_dir + '/' + run_name + '/gnn_checkpoints.pth'
    create_dir(config.train_config.checkpoint_dir + '/' + run_name)
    log_dir = config.train_config.tensorboard_dir + '/' + run_name
    writer = SummaryWriter(log_dir=log_dir)

    # global nb_iter_batch
    nb_iter_batch = [0]

    ## We create the dataset before the loop starts. Use the same dataset for every run, 
    #  effectively turning 'runs' into 'epochs'
    # run_dataset = create_dataset(config.train_config.samples_per_run, config.data_config,
    #                                  run=0, matlab_engine=matlab_engine)

    for run in range(config.train_config.num_runs):
        # we create the data before the training loop starts for efficiency,
        # at the loop we only slice batches and convert to tensors
        run_dataset = create_dataset(config.train_config.samples_per_run, config.data_config,
                                     run=run, matlab_engine=matlab_engine)

        train_run(run_dataset, run, batch_size, config,
                               model, optimizer,
                               checkpoint_prefix,
                               eval_dataset, eval_dataset_dgl, eval_config,
                               device, nb_iter_batch, writer, scheduler)


    if config.train_config.coarsen:
        old_model = copy.deepcopy(model)

        for run in range(config.train_config.num_runs):
            run_dataset = create_dataset(config.train_config.samples_per_run, config.data_config,
                                         run=run, matlab_engine=matlab_engine)

            fine_data_config = copy.deepcopy(config.data_config)
            # RS coarsens to roughly 1/3 of the size of the grid, CLJP to roughly 1/2
            fine_data_config.num_unknowns = config.data_config.num_unknowns * 2
            fine_run_dataset = create_dataset(config.train_config.samples_per_run,
                                              fine_data_config,
                                              run=run,
                                              matlab_engine=matlab_engine)
            coarse_run_dataset = create_coarse_dataset(fine_run_dataset, old_model,
                                                       config.data_config,
                                                       config.run_config,
                                                       matlab_engine=matlab_engine)

            combined_run_dataset = run_dataset + coarse_run_dataset
            combined_run_dataset = combined_run_dataset.shuffle()

            train_run(combined_run_dataset, run, batch_size, config,
                                   model, optimizer,
                                   checkpoint_prefix,
                                   eval_dataset, eval_dataset_dgl, eval_config,
                                   device, nb_iter_batch, writer, scheduler)


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    fire.Fire(train)
