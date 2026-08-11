"""Neural cellular automata on graphs."""
from gnca import damage
from gnca.graphs import knn_graph, random_geometric_graph, watts_strogatz_graph
from gnca.inference import Checkpoint, load_checkpoint, rollout
from gnca.model import GraphNCA, alive_mask, load_rule, seed_state

__all__ = [
    "Checkpoint",
    "GraphNCA",
    "alive_mask",
    "damage",
    "knn_graph",
    "load_checkpoint",
    "load_rule",
    "random_geometric_graph",
    "rollout",
    "seed_state",
    "watts_strogatz_graph",
]
