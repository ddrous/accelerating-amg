import logging
import pathlib
import pickle
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import flax
import flax.linen as nn
from absl import app, flags
from jraph.ogb_examples import data_utils

class EncodeProcessDecodeNonRecurrent(nn.Module):
    """
    similar to EncodeProcessDecode, but with non-recurrent core
    see docs for EncodeProcessDecode
    """

    num_cores: int = 3,
    edge_output_size: int = None,
    node_output_size: int = None,
    global_output_size: int = None,
    global_block: bool = True,
    latent_size: int = 16,
    num_layers: int = 2,
    concat_encoder: bool = True,
    name: str = "EncodeProcessDecodeNonRecurrent"

    def setup(self):
        self._encoder = MLPGraphIndependent(latent_size=self.latent_size, num_layers=self.num_layers)
        self._cores = [MLPGraphNetwork(latent_size=self.latent_size, num_layers=self.num_layers,
                                       global_block=self.global_block) for _ in range(self.num_cores)]
        self._decoder = MLPGraphIndependent(latent_size=self.latent_size, num_layers=self.num_layers)

        # Transforms the outputs into the appropriate shapes.
        if self.edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: nn.Dense(self.edge_output_size, name="edge_output")
        if self.node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: nn.Dense(self.node_output_size, name="node_output")
        if self.global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: nn.Dense(self.global_output_size, name="global_output")

        self._output_transform = GraphIndependent(edge_fn, node_fn, global_fn)

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


class MLPGraphNetwork(nn.Module):
    """GraphNetwork with MLP edge, node, and global models."""

    latent_size: int = 16, 
    num_layers: int = 2, 
    global_block: int = True, 
    last_round: int = False,
    name: str = "MLPGraphNetwork"

    def setup(self):
        partial_make_mlp_model = partial(make_mlp_model, latent_size=self.latent_size, num_layers=self.num_layers,
                                         last_round_edges=False)
        if self.last_round:
            partial_make_mlp_model_edges = partial(make_mlp_model, latent_size=self.latent_size, num_layers=self.num_layers,
                                                   last_round_edges=True)
        else:
            partial_make_mlp_model_edges = partial_make_mlp_model

        if self.global_block:
            self._network = jraph.GraphNetwork(partial_make_mlp_model_edges, 
                                                    partial_make_mlp_model,
                                                    partial_make_mlp_model,
                                                    aggregate_edges_for_nodes_fn = jraph.segment_mean,
                                                    aggregate_edges_for_globals_fn = jraph.segment_mean,
                                                    aggregate_nodes_for_globals_fn = jraph.segment_mean)
        else:
            self._network = jraph.GraphNetwork(partial_make_mlp_model_edges, 
                                                partial_make_mlp_model,
                                                make_identity_model,
                                                aggregate_edges_for_nodes_fn = jraph.segment_mean,
                                                aggregate_edges_for_globals_fn = jraph.segment_mean,
                                                aggregate_nodes_for_globals_fn = jraph.segment_mean)

    def __call__(self, inputs):
        return self._network(inputs)


class MLPGraphIndependent(nn.Module):
    """GraphIndependent with MLP edge, node, and global models."""

    latent_size: int = 16 
    num_layers: int = 2
    name: str = "MLPGraphIndependent"

    def setup(self):
        partial_make_mlp_model = partial(make_mlp_model, 
                                            latent_size=self.latent_size, 
                                            num_layers=self.num_layers,
                                            last_round_edges=False)

        self._network = GraphIndependent(
            edge_model_fn=partial_make_mlp_model,
            node_model_fn=partial_make_mlp_model,
            global_model_fn=partial_make_mlp_model)

    def __call__(self, inputs:jraph.GraphsTuple) -> jraph.GraphsTuple:
        return self._network(inputs)

class GraphIndependent(nn.Module):
  """A graph block that applies models to the graph elements independently.
  The inputs and outputs are graphs. The corresponding models are applied to
  each element of the graph (edges, nodes and globals) in parallel and
  independently of the other elements. It can be used to encode or
  decode the elements of a graph.
  """

  edge_model_fn: callable = None,
  node_model_fn: callable = None,
  global_model_fn: callable = None,
  name: str = "graph_independent"

  def setup(self):
    # The use of snt.Module below is to ensure the ops and variables that
    # result from the edge/node/global_model_fns are scoped analogous to how
    # the Edge/Node/GlobalBlock classes do.
    if self.edge_model_fn is None:
        self._edge_model = lambda x: x
    else:
        self._edge_model = WrappedModelFnModule(
            self.edge_model_fn, name="edge_model")
    if self.node_model_fn is None:
        self._node_model = lambda x: x
    else:
        self._node_model = WrappedModelFnModule(
            self.node_model_fn, name="node_model")
    if self.global_model_fn is None:
        self._global_model = lambda x: x
    else:
        self._global_model = WrappedModelFnModule(
            self.global_model_fn, name="global_model")

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
    return graph._replace(
        edges=self._edge_model(graph.edges, **edge_model_kwargs),
        nodes=self._node_model(graph.nodes, **node_model_kwargs),
        globals=self._global_model(graph.globals, **global_model_kwargs))


class WrappedModelFnModule(nn.Module):
  """Wraps a model_fn as a Sonnet module with a name.
  Following `blocks.py` convention, a `model_fn` is a callable that, when called
  with no arguments, returns a callable similar to a Sonnet module instance.
  """
  model_fn: callable = None 
  name: str = None

  def setup(self):
    self._model = self.model_fn()

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
        return nn.Sequential([nn.Dense(latent_size), jax.nn.relu]* num_layers + [nn.Dense(1)])
    else:
        return nn.Sequential([nn.Dense(latent_size), jax.nn.relu]* (num_layers-1) + [nn.Dense(latent_size)]) ## TODO Check back here !

class IdentityModule(nn.Module):
    def __call__(self, x: jraph.GraphsTuple) -> jraph.GraphsTuple:
        return x

def make_identity_model():
    return IdentityModule()
