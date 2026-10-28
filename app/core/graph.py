from dataclasses import dataclass, field
from typing import Any, Protocol
import networkx as nx


@dataclass
class GraphFeatureVector:
    """
    Topological features extracted from an applicant's bounded 2-hop ego-network.
    Standardized contract for M2 (Graph-enhanced LightGBM) and M3 (Calibrated Fusion).
    """
    shared_phone_degree: float = 0.0
    shared_device_degree: float = 0.0
    shared_ssn_degree: float = 0.0
    ego_node_count: float = 1.0
    ego_edge_count: float = 0.0
    ego_density: float = 0.0
    max_neighbor_risk: float = 0.0
    cycle_count: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "shared_phone_degree": self.shared_phone_degree,
            "shared_device_degree": self.shared_device_degree,
            "shared_ssn_degree": self.shared_ssn_degree,
            "ego_node_count": self.ego_node_count,
            "ego_edge_count": self.ego_edge_count,
            "ego_density": self.ego_density,
            "max_neighbor_risk": self.max_neighbor_risk,
            "cycle_count": self.cycle_count,
        }


@dataclass
class GraphSubnetwork:
    """
    Serializable container for a bounded transient ego-network subnetwork.
    """
    root_id: str
    depth: int = 2
    graph: nx.Graph = field(default_factory=nx.Graph)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


class IdentityGraphEngine(Protocol):
    """
    Domain protocol for identity graph operations.
    Guarantee: Only loads bounded 2-hop ego-networks from PostgreSQL;
    never holds an unconstrained global graph in process memory.
    """
    def add_identity_node(self, applicant_id: str, attributes: dict[str, Any]) -> None:
        ...

    def find_linked_clusters(self, applicant_id: str, depth: int = 2) -> GraphSubnetwork:
        ...

    def compute_graph_features(self, applicant_id: str) -> GraphFeatureVector:
        ...
