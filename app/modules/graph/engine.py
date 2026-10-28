import logging
from typing import Any
import networkx as nx
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.graph import GraphFeatureVector, GraphSubnetwork, IdentityGraphEngine
from app.core.security import hash_pii
from app.modules.graph.models import (
    HEX_64_PATTERN,
    ApplicantEdge,
    ApplicantNode,
    EdgeType,
    NodeType,
)

logger = logging.getLogger(__name__)


def _to_pseudonymized_hash(value: Any, is_numeric_id: bool = False) -> str | None:
    if value is None:
        return None
    val_str = str(value).strip()
    if not val_str:
        return None
    if HEX_64_PATTERN.match(val_str):
        return val_str.lower()
    return hash_pii(val_str, is_numeric_id=is_numeric_id)


class RelationalIdentityGraphEngine(IdentityGraphEngine):
    """
    PostgreSQL/SQLite-backed relational identity graph engine.
    Extracts strictly bounded 2-hop ego-networks into transient NetworkX graphs,
    computing topological features without global in-memory graph bloat.
    """

    def __init__(
        self,
        db: Session,
        max_neighbors_per_hop: int = 100,
        max_total_edges: int = 500,
    ) -> None:
        self.db = db
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self.max_total_edges = max_total_edges

    def add_identity_node(self, applicant_id: str, attributes: dict[str, Any]) -> None:
        """
        Ingests an applicant node and establishes directional identity edges.
        Automatically pseudonymizes raw PII (Aadhaar, phone, email, device) to HMAC-SHA256.
        """
        phone_hash = _to_pseudonymized_hash(attributes.get("phone"), is_numeric_id=True)
        email_hash = _to_pseudonymized_hash(attributes.get("email"))
        identity_hash = _to_pseudonymized_hash(
            attributes.get("identity_number") or attributes.get("aadhaar") or attributes.get("ssn"),
            is_numeric_id=True,
        )
        device_hash = _to_pseudonymized_hash(
            attributes.get("device_fingerprint") or attributes.get("device_hash")
        )
        ip_hash = _to_pseudonymized_hash(attributes.get("ip_address"))

        risk_score = None
        if "risk_score" in attributes and attributes["risk_score"] is not None:
            risk_score = float(attributes["risk_score"])

        is_fraud = attributes.get("is_fraud")
        status = attributes.get("status", "pending")
        properties = attributes.get("properties") or {}

        # 1. Upsert ApplicantNode
        node = self.db.scalar(
            select(ApplicantNode).where(ApplicantNode.applicant_id == applicant_id)
        )
        if node is None:
            node = ApplicantNode(
                applicant_id=applicant_id,
                node_type=NodeType.APPLICANT.value,
                identity_hash=identity_hash,
                phone_hash=phone_hash,
                email_hash=email_hash,
                device_fingerprint_hash=device_hash,
                ip_address_hash=ip_hash,
                status=status,
                risk_score=risk_score,
                is_fraud=is_fraud,
                properties_json=properties,
            )
            self.db.add(node)
        else:
            if identity_hash:
                node.identity_hash = identity_hash
            if phone_hash:
                node.phone_hash = phone_hash
            if email_hash:
                node.email_hash = email_hash
            if device_hash:
                node.device_fingerprint_hash = device_hash
            if ip_hash:
                node.ip_address_hash = ip_hash
            if risk_score is not None:
                node.risk_score = risk_score
            if is_fraud is not None:
                node.is_fraud = is_fraud
            node.status = status
            if properties:
                node.properties_json = {**dict(node.properties_json), **properties}

        # 2. Add relational edges
        edge_specs = [
            (phone_hash, EdgeType.HAS_PHONE.value, NodeType.PHONE.value),
            (email_hash, EdgeType.HAS_EMAIL.value, NodeType.EMAIL.value),
            (identity_hash, EdgeType.HAS_IDENTITY.value, NodeType.IDENTITY.value),
            (device_hash, EdgeType.HAS_DEVICE.value, NodeType.DEVICE.value),
            (ip_hash, EdgeType.HAS_IP.value, NodeType.IP_ADDRESS.value),
        ]

        for target_hash, edge_type, target_type in edge_specs:
            if not target_hash:
                continue
            existing_edge = self.db.scalar(
                select(ApplicantEdge.id).where(
                    ApplicantEdge.source_hash == applicant_id,
                    ApplicantEdge.target_hash == target_hash,
                    ApplicantEdge.edge_type == edge_type,
                )
            )
            if not existing_edge:
                new_edge = ApplicantEdge(
                    source_hash=applicant_id,
                    target_hash=target_hash,
                    edge_type=edge_type,
                    source_type=NodeType.APPLICANT.value,
                    target_type=target_type,
                    weight=1.0,
                    properties_json={},
                )
                self.db.add(new_edge)

        self.db.commit()

    def find_linked_clusters(self, applicant_id: str, depth: int = 2) -> GraphSubnetwork:
        """
        Extracts the bounded 2-hop ego-network around applicant_id from indexed PostgreSQL edges.
        Loads only the localized subgraph into a transient NetworkX Graph and releases memory.
        """
        # Iterative BFS up to specified depth
        visited_nodes: set[str] = {applicant_id}
        current_frontier: set[str] = {applicant_id}
        all_edges_dict: dict[tuple[str, str, str], ApplicantEdge] = {}

        for _ in range(depth):
            if not current_frontier or len(all_edges_dict) >= self.max_total_edges:
                break

            limit_per_step = min(self.max_neighbors_per_hop * len(current_frontier), self.max_total_edges)
            step_stmt = (
                select(ApplicantEdge)
                .where(
                    or_(
                        ApplicantEdge.source_hash.in_(current_frontier),
                        ApplicantEdge.target_hash.in_(current_frontier),
                    )
                )
                .limit(limit_per_step)
            )
            step_edges = list(self.db.scalars(step_stmt).all())

            next_frontier: set[str] = set()
            for edge in step_edges:
                all_edges_dict[(edge.source_hash, edge.target_hash, edge.edge_type)] = edge
                if edge.source_hash not in visited_nodes:
                    next_frontier.add(edge.source_hash)
                if edge.target_hash not in visited_nodes:
                    next_frontier.add(edge.target_hash)

            visited_nodes.update(next_frontier)
            current_frontier = next_frontier

        if not all_edges_dict:
            # Check if root applicant node exists as an isolated node
            root_node = self.db.scalar(
                select(ApplicantNode).where(ApplicantNode.applicant_id == applicant_id)
            )
            G = nx.Graph()
            if root_node:
                G.add_node(
                    applicant_id,
                    type="applicant",
                    risk=float(root_node.risk_score or 0.0),
                    status=root_node.status,
                    is_fraud=root_node.is_fraud,
                )
                return GraphSubnetwork(
                    root_id=applicant_id,
                    depth=depth,
                    graph=G,
                    nodes=[{"id": applicant_id, **G.nodes[applicant_id]}],
                    edges=[],
                )
            return GraphSubnetwork(root_id=applicant_id, depth=depth, graph=G, nodes=[], edges=[])

        # Collect all node identifiers in this subnetwork
        all_node_hashes: set[str] = {applicant_id}
        for edge in all_edges_dict.values():
            all_node_hashes.add(edge.source_hash)
            all_node_hashes.add(edge.target_hash)

        # Batch-fetch metadata for applicant nodes
        applicant_nodes = list(
            self.db.scalars(
                select(ApplicantNode).where(ApplicantNode.applicant_id.in_(all_node_hashes))
            ).all()
        )
        app_by_id = {node.applicant_id: node for node in applicant_nodes}

        # Build transient NetworkX graph
        G = nx.Graph()

        for node_hash in all_node_hashes:
            if node_hash in app_by_id:
                app = app_by_id[node_hash]
                G.add_node(
                    node_hash,
                    type=NodeType.APPLICANT.value,
                    risk=float(app.risk_score or 0.0),
                    status=app.status,
                    is_fraud=app.is_fraud,
                )
            else:
                # Infer identifier type from the connected edges
                node_type = "identifier"
                for edge in all_edges_dict.values():
                    if edge.target_hash == node_hash:
                        node_type = edge.target_type
                        break
                    elif edge.source_hash == node_hash:
                        node_type = edge.source_type
                        break
                G.add_node(
                    node_hash,
                    type=node_type,
                    risk=0.0,
                    status="active",
                )

        for edge in all_edges_dict.values():
            G.add_edge(
                edge.source_hash,
                edge.target_hash,
                edge_type=edge.edge_type,
                weight=edge.weight,
            )

        nodes_list = [{"id": n, **G.nodes[n]} for n in G.nodes]
        edges_list = [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges]

        return GraphSubnetwork(
            root_id=applicant_id,
            depth=depth,
            graph=G,
            nodes=nodes_list,
            edges=edges_list,
        )

    def compute_graph_features(self, applicant_id: str) -> GraphFeatureVector:
        """
        Computes the 8 topological fraud features on the bounded 2-hop ego-network.
        Max neighbor risk excludes applicant's own risk score to prevent target leakage.
        """
        subnetwork = self.find_linked_clusters(applicant_id, depth=2)
        G = subnetwork.graph

        if applicant_id not in G:
            return GraphFeatureVector(ego_node_count=1.0)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        # Shared attribute degrees: count distinct other applicants reachable via each identifier type
        phone_neighbors = [
            n for n in G.neighbors(applicant_id)
            if G.nodes[n].get("type") in ("phone", NodeType.PHONE.value)
        ]
        other_applicants_phone: set[str] = set()
        for p in phone_neighbors:
            other_applicants_phone.update(
                a for a in G.neighbors(p)
                if a != applicant_id and G.nodes[a].get("type") in ("applicant", NodeType.APPLICANT.value)
            )

        device_neighbors = [
            n for n in G.neighbors(applicant_id)
            if G.nodes[n].get("type") in ("device", NodeType.DEVICE.value)
        ]
        other_applicants_device: set[str] = set()
        for d in device_neighbors:
            other_applicants_device.update(
                a for a in G.neighbors(d)
                if a != applicant_id and G.nodes[a].get("type") in ("applicant", NodeType.APPLICANT.value)
            )

        identity_neighbors = [
            n for n in G.neighbors(applicant_id)
            if G.nodes[n].get("type") in ("identity", "ssn", NodeType.IDENTITY.value)
        ]
        other_applicants_identity: set[str] = set()
        for i in identity_neighbors:
            other_applicants_identity.update(
                a for a in G.neighbors(i)
                if a != applicant_id and G.nodes[a].get("type") in ("applicant", NodeType.APPLICANT.value)
            )

        density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        # Maximum neighbor risk strictly among other applicants
        neighbor_risks = [
            G.nodes[n].get("risk", 0.0)
            for n in G.nodes
            if n != applicant_id and G.nodes[n].get("type") in ("applicant", NodeType.APPLICANT.value)
        ]
        max_risk = float(max(neighbor_risks)) if neighbor_risks else 0.0

        # Cycle count via cycle basis on the ego-network
        cycle_count = float(len(nx.cycle_basis(G)))

        return GraphFeatureVector(
            shared_phone_degree=float(len(other_applicants_phone)),
            shared_device_degree=float(len(other_applicants_device)),
            shared_ssn_degree=float(len(other_applicants_identity)),
            ego_node_count=float(n_nodes),
            ego_edge_count=float(n_edges),
            ego_density=round(density, 4),
            max_neighbor_risk=round(max_risk, 4),
            cycle_count=cycle_count,
        )

    def get_1hop_identifier_degree(self, target_hash: str, edge_type: str) -> int:
        """
        Fast synchronous 1-hop degree check.
        Queries indexed applicant_edges on (target_hash, edge_type) in O(log n).
        """
        stmt = (
            select(ApplicantEdge.id)
            .where(
                ApplicantEdge.target_hash == target_hash,
                ApplicantEdge.edge_type == edge_type,
            )
        )
        return len(list(self.db.scalars(stmt).all()))
