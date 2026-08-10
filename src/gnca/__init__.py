"""Neural cellular automata on graphs."""
from gnca import damage
from gnca.graphs import knn_graph, random_geometric_graph, watts_strogatz_graph
from gnca.model import GraphNCA, alive_mask, load_rule, seed_state
from gnca.targets import heart_target, ring_target

__all__ = [
    "GraphNCA",
    "alive_mask",
    "damage",
    "heart_target",
    "knn_graph",
    "load_rule",
    "random_geometric_graph",
    "ring_target",
    "seed_state",
    "watts_strogatz_graph",
]
