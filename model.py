import jraph
import matlab
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental.sparse import BCOO
import flax
import optax
import flax.linen as nn
from flax.training import train_state, checkpoints
import pyamg
from clu import parameter_overview

from data import As_poisson_grid
# from jraph_model import EncodeProcessDecodeNonRecurrent
from flax_model import EncodeProcessDecodeNonRecurrent
from utils import print_available_gpu
import copy


def clone_model(model, params, model_config, run_config, matlab_engine):
    clone = create_model(model_config)
    _ = init_model(clone, run_config, matlab_engine)

    return clone, copy.deepcopy(params)



def get_model(model_config, run_config, train_config, matlab_engine, train=False):
    # dummy_input = As_poisson_grid(1, 7 ** 2)[0]
    # checkpoint_dir = './training_dir/' + model_name
    graph_model, params, optimiser = load_model(train_config, model_config,
                                                     run_config,
                                                     matlab_engine, get_optimizer=train)
    if train:
        return graph_model, params, optimiser
    else:
        return graph_model, params


def init_model(model, run_config, matlab_engine):
    dummy_input = pyamg.gallery.poisson((7, 7), type='FE', format='csr')
    dummy_graph_tuple = csrs_to_graphs_tuple([dummy_input], 
                                            matlab_engine, 
                                            coarse_nodes_list=np.array([[0, 1]]),
                                            baseline_P_list=[jnp.array(dummy_input.toarray()[:, [0, 1]])],
                                            node_indicators=run_config.node_indicators,
                                            edge_indicators=run_config.edge_indicators)
    ## Randomly initialise the weights
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    params = model.init(key, dummy_graph_tuple)

    ## Some printing
    model.tabulate(key, dummy_graph_tuple)
    parameter_overview.log_parameter_overview(params)

    return params

def load_model(train_config, model_config, run_config, matlab_engine, get_optimizer=False):
    model = create_model(model_config)

    params = init_model(model, run_config, matlab_engine)

    decay_steps = 100
    decay_rate = 1.0
    learning_rate = optax.exponential_decay(init_value=train_config.learning_rate, 
                                            transition_steps=decay_steps, 
                                            decay_rate=decay_rate)
    optimiser = optax.adam(learning_rate=learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply,
                                        params=params,
                                        tx=optimiser)

    checkpoint_dir = train_config.checkpoint_dir
    latest_checkpoint = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=state)

    if latest_checkpoint is None:
        raise RuntimeError(f'training_dir {checkpoint_dir} does not exist')

    return model, latest_checkpoint.params, optimiser


def create_model(model_config):
    # return EncodeProcessDecodeNonRecurrent(num_cores=model_config.mp_rounds, edge_output_size=1,
    #                                         node_output_size=1, global_block=model_config.global_block,
    #                                         latent_size=model_config.latent_size,
    #                                         num_layers=model_config.mlp_layers,
    #                                         concat_encoder=model_config.concat_encoder)
    return EncodeProcessDecodeNonRecurrent(
                                          latent_size=model_config.latent_size,
                                          num_mlp_layers=model_config.mlp_layers,
                                          message_passing_steps=model_config.mp_rounds,
                                          output_size=1,
                                          dropout_rate=0,
                                          skip_connections=model_config.concat_encoder,
                                          layer_norm=True,
                                          use_edge_model=True,
                                          deterministic=True)


def csrs_to_graphs_tuple(csrs, matlab_engine, node_feature_size=128, coarse_nodes_list=None, baseline_P_list=None,
                         node_indicators=True, edge_indicators=True):
    dtype = jnp.float32

    # build up the arguments for the GraphsTuple constructor
    n_node = jnp.array([csr.shape[0] for csr in csrs])
    n_edge = jnp.array([csr.nnz for csr in csrs])

    if not edge_indicators:
        numpy_edges = np.concatenate([csr.data for csr in csrs])
        edges = jnp.expand_dims(jnp.array(numpy_edges, dtype=dtype), axis=1)
    else:
        edge_encodings_list = []
        for csr, coarse_nodes, baseline_P in zip(csrs, coarse_nodes_list, baseline_P_list):
            if isinstance(baseline_P, jax.numpy.ndarray):
                baseline_P = csr_matrix(np.array(baseline_P))

            baseline_P_rows, baseline_P_cols = P_square_sparsity_pattern(baseline_P, baseline_P.shape[0],
                                                                         coarse_nodes, matlab_engine)
            coo = csr.tocoo()

            # construct numpy structured arrays, where each element is a tuple (row,col), so that we can later use
            # the numpy set function in1d()
            baseline_P_indices = np.core.records.fromarrays([baseline_P_rows, baseline_P_cols], dtype='i,i')
            coo_indices = np.core.records.fromarrays([coo.row, coo.col], dtype='i,i')

            same_indices = np.in1d(coo_indices, baseline_P_indices, assume_unique=True)
            baseline_edges = same_indices.astype(np.float64)
            non_baseline_edges = (~same_indices).astype(np.float64)

            edge_encodings = np.stack([coo.data, baseline_edges, non_baseline_edges]).T
            edge_encodings_list.append(edge_encodings)
        numpy_edges = np.concatenate(edge_encodings_list)
        edges = jnp.array(numpy_edges, dtype=dtype)

    # COO format for sparse matrices contains a list of row indices and a list of column indices
    coos = [csr.tocoo() for csr in csrs]
    senders_numpy = np.concatenate([coo.row for coo in coos])
    senders = jnp.array(senders_numpy)
    receivers_numpy = np.concatenate([coo.col for coo in coos])
    receivers = jnp.array(receivers_numpy)

    # # see the source of _concatenate_data_dicts for explanation
    # offsets = gn.utils_tf._compute_stacked_offsets(n_node, n_edge)
    # senders += offsets
    # receivers += offsets

    if not node_indicators:
        nodes = None
    else:
        node_encodings_list = []
        graphs_list = []
        for csr, coarse_nodes in zip(csrs, coarse_nodes_list):
            coarse_indices = np.in1d(range(csr.shape[0]), coarse_nodes, assume_unique=True)

            coarse_node_encodings = coarse_indices.astype(np.float64)
            fine_node_encodings = (~coarse_indices).astype(np.float64)
            node_encodings = np.stack([coarse_node_encodings, fine_node_encodings]).T

            node_encodings_list.append(node_encodings)

        numpy_nodes = np.concatenate(node_encodings_list)
        nodes = jnp.array(numpy_nodes, dtype=dtype)

        # COO format for sparse matrices contains a list of row indices and a list of column indices
        coo = csr.tocoo()
        senders = jnp.array(coo.row)
        receivers = jnp.array(coo.col)

        graph_tuple = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=None,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge
        )
        graphs_list.append(graph_tuple)

    graphs_tuple = jraph.batch(graphs_list)   ## TODO Batch-size is 1, so nothing occurs. Don't use batch as it doesn't support jit

    if not node_indicators:
        graphs_tuple = set_zero_node_features(graphs_tuple, 1, dtype=dtype)

    graphs_tuple = set_zero_global_features(graphs_tuple, node_feature_size, dtype=dtype)

    return graphs_tuple


def P_square_sparsity_pattern(P, size, coarse_nodes, matlab_engine):
    # P_coo = BCOO.fromdense(P)
    # P_rows = matlab.double((np.array(P_coo.indices[:,0], dtype=np.float64) + 1))
    # P_cols = matlab.double((np.array(P_coo.indices[:,1], dtype=np.float64) + 1))
    # P_values = matlab.double(np.array(P_coo.data, dtype=np.float64))
    # coarse_nodes = matlab.double((np.array(coarse_nodes, dtype=np.float64) + 1))
    P_coo = P.tocoo()
    P_rows = matlab.double((np.float64(P_coo.row) + 1))
    P_cols = matlab.double((np.float64(P_coo.col) + 1))
    P_values = matlab.double(np.float64(P_coo.data))
    coarse_nodes = matlab.double((np.float64(coarse_nodes) + 1))
    rows, cols = matlab_engine.square_P(P_rows, P_cols, P_values, size, coarse_nodes,  nargout=2)
    rows = np.array(rows._data).reshape(rows.size, order='F') - 1
    cols = np.array(cols._data).reshape(cols.size, order='F') - 1
    rows, cols = rows.T[0], cols.T[0]
    return rows, cols


def graphs_tuple_to_sparse_tensor(graphs_tuple):
    senders = graphs_tuple.senders
    receivers = graphs_tuple.receivers
    indices = tf.cast(tf.stack([senders, receivers], axis=1), tf.int64)

    # first element in the edge feature is the value, the other elements are metadata
    values = jnp.squeeze(graphs_tuple.edges[:, 0])

    shape = jnp.concatenate([graphs_tuple.n_node, graphs_tuple.n_node], axis=0)
    # shape = tf.cast(shape, tf.int64)

    matrix_uo = BCOO((values, indices), shape=shape)
    # reordering is required because the pyAMG coarsening step does not preserve indices order

    matrix = BCOO.fromdense(matrix_uo.todense())    ### TODO dirty trick to reorder indices

    return matrix


def to_prolongation_matrix_csr(matrix, coarse_nodes, baseline_P, nodes, normalize_rows=True,
                               normalize_rows_by_node=False):
    """
    sparse version of the above function, for when the dense matrix is too large to fit in GPU memory
    used only for inference, so no need for backpropagation, inputs are csr matrices
    """
    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    matrix.setdiag(np.ones(matrix.shape[0]))

    # select only columns corresponding to coarse nodes
    matrix = matrix[:, coarse_nodes]

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
    return matrix


def to_prolongation_matrix_tensor(matrix, coarse_nodes, baseline_P, nodes,
                                  normalize_rows=True,
                                  normalize_rows_by_node=False):
    dtype = jnp.float64
    matrix = matrix.todense()
    matrix = matrix.astype(dtype)

    # prolongation from coarse point to itself should be identity. This corresponds to 1's on the diagonal
    diag_ind = jnp.diag_indices(matrix.shape[0])
    matrix = matrix.at[diag_ind].set(jnp.ones(matrix.shape[0], dtype=dtype))

    # select only columns corresponding to coarse nodes
    matrix = matrix[:, coarse_nodes]

    # set sparsity pattern (interpolatory sets) to be of baseline prolongation
    baseline_zero_mask = jnp.not_equal(baseline_P, jnp.zeros_like(baseline_P)).astype(dtype)
    # print("Second check:", matrix.shape, baseline_zero_mask.shape)
    matrix = matrix * baseline_zero_mask

    if normalize_rows:
        if normalize_rows_by_node:
            baseline_row_sum = nodes
        else:
            baseline_row_sum = jnp.sum(baseline_P, axis=1)
        baseline_row_sum = baseline_row_sum.astype(dtype)

        # matrix_row_sum = jnp.sum(matrix, axis=1)
        # matrix_row_sum = matrix_row_sum.astype(dtype)

        # # there might be a few rows that are all 0's - corresponding to fine points that are not connected to any
        # # coarse point. We use "divide_no_nan" to let these rows remain 0's
        # matrix_row_sum = jnp.where(matrix_row_sum==0.0, 1.0, matrix_row_sum)
        # matrix = jnp.divide(matrix, jnp.reshape(matrix_row_sum, (-1, 1)))
        # matrix = jnp.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        matrix = matrix * jnp.reshape(baseline_row_sum, (-1, 1))
    return matrix


def graphs_tuple_to_sparse_matrices(graphs_tuple, return_nodes=False):
    # graphs = graphs_tuple.unbatch(graphs_tuple)   ### TODO Always unbatch in a implicit way that allows jit
    graphs = [graphs_tuple]

    matrices = [graphs_tuple_to_sparse_tensor(graph) for graph in graphs]

    if return_nodes:
        nodes_list = [jnp.squeeze(graph.nodes) for graph in graphs]
        return matrices, nodes_list
    else:
        return matrices



#### A bunch of functions copied from graphs_net: https://github.com/deepmind/graph_nets/blob/master/graph_nets/utils_tf.py 
def set_zero_node_features(graph,
                           node_size,
                           dtype=tf.float32,
                           name="set_zero_node_features"):
  """Completes the node state of a graph.
  Args:
    graph: A `graphs.GraphsTuple` with a `None` edge state.
    node_size: (int) the dimension for the created node features.
    dtype: (tensorflow type) the type for the created nodes features.
    name: (string, optional) A name for the operation.
  Returns:
    The same graph but for the node field, which is a `Tensor` of shape
    `[number_of_nodes, node_size]`  where `number_of_nodes = sum(graph.n_node)`,
    with type `dtype`, filled with zeros.
  Raises:
    ValueError: If the `NODES` field is not None in `graph`.
    ValueError: If `node_size` is None.
  """
  if graph.nodes is not None:
    raise ValueError(
        "Cannot complete node state if the graph already has node features.")
  if node_size is None:
    raise ValueError("Cannot complete nodes with None node_size")
  with tf.name_scope(name):
    n_nodes = tf.reduce_sum(graph.n_node)
    return graph._replace(
        nodes=tf.zeros(shape=[n_nodes, node_size], dtype=dtype))


def set_zero_edge_features(graph,
                           edge_size,
                           dtype=tf.float32,
                           name="set_zero_edge_features"):
  """Completes the edge state of a graph.
  Args:
    graph: A `graphs.GraphsTuple` with a `None` edge state.
    edge_size: (int) the dimension for the created edge features.
    dtype: (tensorflow type) the type for the created edge features.
    name: (string, optional) A name for the operation.
  Returns:
    The same graph but for the edge field, which is a `Tensor` of shape
    `[number_of_edges, edge_size]`, where `number_of_edges = sum(graph.n_edge)`,
    with type `dtype` and filled with zeros.
  Raises:
    ValueError: If the `EDGES` field is not None in `graph`.
    ValueError: If the `RECEIVERS` or `SENDERS` field are None in `graph`.
    ValueError: If `edge_size` is None.
  """
  if graph.edges is not None:
    raise ValueError(
        "Cannot complete edge state if the graph already has edge features.")
  if graph.receivers is None or graph.senders is None:
    raise ValueError(
        "Cannot complete edge state if the receivers or senders are None.")
  if edge_size is None:
    raise ValueError("Cannot complete edges with None edge_size")
  with tf.name_scope(name):
    senders_leading_size = graph.senders.shape.as_list()[0]
    if senders_leading_size is not None:
      n_edges = senders_leading_size
    else:
      n_edges = tf.reduce_sum(graph.n_edge)
    return graph._replace(
        edges=tf.zeros(shape=[n_edges, edge_size], dtype=dtype))


def set_zero_global_features(graph,
                             global_size,
                             dtype=tf.float32,
                             name="set_zero_global_features"):
  """Completes the global state of a graph.
  Args:
    graph: A `graphs.GraphsTuple` with a `None` global state.
    global_size: (int) the dimension for the created global features.
    dtype: (tensorflow type) the type for the created global features.
    name: (string, optional) A name for the operation.
  Returns:
    The same graph but for the global field, which is a `Tensor` of shape
    `[num_graphs, global_size]`, type `dtype` and filled with zeros.
  Raises:
    ValueError: If the `GLOBALS` field of `graph` is not `None`.
    ValueError: If `global_size` is not `None`.
  """
  if graph.globals is not None:
    raise ValueError(
        "Cannot complete global state if graph already has global features.")
  if global_size is None:
    raise ValueError("Cannot complete globals with None global_size")
  with jax.named_scope(name):
    n_graphs = list(graph.n_node.shape)[0]
    return graph._replace(
        globals=tf.zeros(shape=[n_graphs, global_size], dtype=dtype))


def get_num_graphs(input_graphs, name="get_num_graphs"):
  """Returns the number of graphs (i.e. the batch size) in `input_graphs`.
  Args:
    input_graphs: A `graphs.GraphsTuple` containing tensors.
    name: (string, optional) A name for the operation.
  Returns:
    An `int` (if a static number of graphs is defined) or a `tf.Tensor` (if the
      number of graphs is dynamic).
  """
  with tf.name_scope(name):
    return _get_shape(input_graphs.n_node)[0]

def _get_shape(tensor):
  """Returns the tensor's shape.
   Each shape element is either:
   - an `int`, when static shape values are available, or
   - a `tf.Tensor`, when the shape is dynamic.
  Args:
    tensor: A `tf.Tensor` to get the shape of.
  Returns:
    The `list` which contains the tensor's shape.
  """

  shape_list = list(tensor.shape)
  if all(s is not None for s in shape_list):
    return shape_list
  shape_tensor = tf.shape(tensor)
  return [shape_tensor[i] if s is None else s for i, s in enumerate(shape_list)]