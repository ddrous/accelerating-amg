import logging
import pathlib
import pickle
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from absl import app, flags
from jraph.ogb_examples import data_utils


class EncodeProcessDecodeNonRecurrent(hk.Module):
    """
    similar to EncodeProcessDecode, but with non-recurrent core
    see docs for EncodeProcessDecode
    """

    def __init__(self,
                 num_cores=3,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 global_block=True,
                 latent_size=16,
                 num_layers=2,
                 concat_encoder=True,
                 name="EncodeProcessDecodeNonRecurrent"):
        super(EncodeProcessDecodeNonRecurrent, self).__init__(name=name)
        self._encoder = MLPGraphIndependent(latent_size=latent_size, num_layers=num_layers)
        self._cores = [MLPGraphNetwork(latent_size=latent_size, num_layers=num_layers,
                                       global_block=global_block) for _ in range(num_cores)]
        self._decoder = MLPGraphIndependent(latent_size=latent_size, num_layers=num_layers)
        self.concat_encoder = concat_encoder

        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: hk.Linear(edge_output_size, name="edge_output")
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: hk.Linear(node_output_size, name="node_output")
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: hk.Linear(global_output_size, name="global_output")

        self._output_transform = hk.GraphIndependent(edge_fn, node_fn, global_fn)

    def __call__(self, input_op):
        latent = self._encoder(input_op)
        # latent0 = jnp.empty((latent.shape[0], 0)) ## TODO to avoid concatenating endcoder results
        latent0 = latent
        for i in range(len(self._cores)):
            if self.concat_encoder and i != 0:
                core_input = jnp.concatenate([latent, latent0], axis=1)
            else:
                core_input = latent
            latent = self._cores[i](core_input)

        return self._output_transform(self._decoder(latent))


class MLPGraphNetwork(hk.Module):
    """GraphNetwork with MLP edge, node, and global models."""

    def __init__(self, latent_size=16, num_layers=2, global_block=True, last_round=False,
                 name="MLPGraphNetwork"):
        super(MLPGraphNetwork, self).__init__(name=name)
        partial_make_mlp_model = partial(make_mlp_model, latent_size=latent_size, num_layers=num_layers,
                                         last_round_edges=False)
        if last_round:
            partial_make_mlp_model_edges = partial(make_mlp_model, latent_size=latent_size, num_layers=num_layers,
                                                   last_round_edges=True)
        else:
            partial_make_mlp_model_edges = partial_make_mlp_model

        if global_block:
            self._network = hk.GraphNetwork(partial_make_mlp_model_edges, partial_make_mlp_model,
                                                    partial_make_mlp_model,
                                                    edge_block_opt={
                                                        "use_globals": True
                                                    },
                                                    node_block_opt={
                                                        "use_globals": True
                                                    },
                                                    global_block_opt={
                                                        "use_globals": True,
                                                        "edges_reducer": tf.unsorted_segment_mean,
                                                        "nodes_reducer": tf.unsorted_segment_mean
                                                    })
        else:
            self._network = hk.GraphNetwork(partial_make_mlp_model_edges, partial_make_mlp_model,
                                                    make_identity_model,
                                                    edge_block_opt={
                                                        "use_globals": False
                                                    },
                                                    node_block_opt={
                                                        "use_globals": False
                                                    },
                                                    global_block_opt={
                                                        "use_globals": False,
                                                    })

    def __call__(self, inputs):
        return self._network(inputs)


class MLPGraphIndependent(hk.Module):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, latent_size=16, num_layers=2, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)

        partial_make_mlp_model = partial(make_mlp_model, 
                                            latent_size=latent_size, 
                                            num_layers=num_layers,
                                            last_round_edges=False)

        self._network = GraphIndependent(
            edge_model_fn=partial_make_mlp_model,
            node_model_fn=partial_make_mlp_model,
            global_model_fn=partial_make_mlp_model)

    def __call__(self, inputs:jraph.GraphsTuple) -> jraph.GraphsTuple:
        return self._network(inputs)

class GraphIndependent(hk.Module):
  """A graph block that applies models to the graph elements independently.
  The inputs and outputs are graphs. The corresponding models are applied to
  each element of the graph (edges, nodes and globals) in parallel and
  independently of the other elements. It can be used to encode or
  decode the elements of a graph.
  """

  def __init__(self,
               edge_model_fn=None,
               node_model_fn=None,
               global_model_fn=None,
               name="graph_independent"):
    """Initializes the GraphIndependent module.
    Args:
      edge_model_fn: A callable that returns an edge model function. The
        callable must return a Sonnet module (or equivalent). If passed `None`,
        will pass through inputs (the default).
      node_model_fn: A callable that returns a node model function. The callable
        must return a Sonnet module (or equivalent). If passed `None`, will pass
        through inputs (the default).
      global_model_fn: A callable that returns a global model function. The
        callable must return a Sonnet module (or equivalent). If passed `None`,
        will pass through inputs (the default).
      name: The module name.
    """
    super(GraphIndependent, self).__init__(name=name)

    # The use of snt.Module below is to ensure the ops and variables that
    # result from the edge/node/global_model_fns are scoped analogous to how
    # the Edge/Node/GlobalBlock classes do.
    if edge_model_fn is None:
        self._edge_model = lambda x: x
    else:
        self._edge_model = WrappedModelFnModule(
            edge_model_fn, name="edge_model")
    if node_model_fn is None:
        self._node_model = lambda x: x
    else:
        self._node_model = WrappedModelFnModule(
            node_model_fn, name="node_model")
    if global_model_fn is None:
        self._global_model = lambda x: x
    else:
        self._global_model = WrappedModelFnModule(
            global_model_fn, name="global_model")

  def __call__(self,
             graph,
             edge_model_kwargs=None,
             node_model_kwargs=None,
             global_model_kwargs=None):
    """Connects the GraphIndependent.
    Args:
      graph: A `graphs.GraphsTuple` containing non-`None` edges, nodes and
        globals.
      edge_model_kwargs: Optional keyword arguments to pass to
        the edge block model.
      node_model_kwargs: Optional keyword arguments to pass to
        the node block model.
      global_model_kwargs: Optional keyword arguments to pass to
        the global block model.
    Returns:
      An output `graphs.GraphsTuple` with updated edges, nodes and globals.
    """
    if edge_model_kwargs is None:
      edge_model_kwargs = {}
    if node_model_kwargs is None:
      node_model_kwargs = {}
    if global_model_kwargs is None:
      global_model_kwargs = {}
    return graph.replace(
        edges=self._edge_model(graph.edges, **edge_model_kwargs),
        nodes=self._node_model(graph.nodes, **node_model_kwargs),
        globals=self._global_model(graph.globals, **global_model_kwargs))


class WrappedModelFnModule(hk.Module):
  """Wraps a model_fn as a Sonnet module with a name.
  Following `blocks.py` convention, a `model_fn` is a callable that, when called
  with no arguments, returns a callable similar to a Sonnet module instance.
  """

  def __init__(self, model_fn, name):
    """Inits the module.
    Args:
      model_fn: callable that, when called with no arguments, returns a callable
          similar to a Sonnet module instance.
      name: Name for the wrapper module.
    """
    super(WrappedModelFnModule, self).__init__(name=name)
    self._model = model_fn()

  def __call__(self, *args, **kwargs):
    return self._model(*args, **kwargs)


# @jraph.concatenated_args
def make_mlp_model(latent_size=16, num_layers=2, last_round_edges=False):
    """Instantiates a new MLP.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Haiku module which contains the MLP.
  """
    if last_round_edges:
        return hk.Sequential([hk.Linear(latent_size), jax.nn.relu]* num_layers + [hk.Linear(1)])
    else:
        return hk.Sequential([hk.Linear(latent_size), jax.nn.relu]* (num_layers-1) + [hk.Linear(latent_size)]) ## TODO Check back here !

class IdentityModule(hk.Module):
    def __call__(self, x: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return x

def make_identity_model():
    return IdentityModule()
