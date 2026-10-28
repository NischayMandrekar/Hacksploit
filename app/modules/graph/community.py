import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
import networkx as nx
from networkx.algorithms import bipartite
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.graph.engine import RelationalIdentityGraphEngine
from app.modules.graph.models import ApplicantNode, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ClusterMember:
    applicant_id: str
    status: str
    risk_score: float
    is_fraud: bool | None
    connected_identifiers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "status": self.status,
            "risk_score": self.risk_score,
            "is_fraud": self.is_fraud,
            "connected_identifiers": self.connected_identifiers,
        }


@dataclass
class CommunitySummary:
    cluster_id: str
    root_applicant_id: str
    total_nodes: int
    total_edges: int
    applicant_members: list[ClusterMember]
    identifier_nodes: list[str]
    density: float
    fraud_prevalence: float
    is_ring: bool
    cycle_count: int
    clique_count: int
    reason_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "root_applicant_id": self.root_applicant_id,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "applicant_members": [m.to_dict() for m in self.applicant_members],
            "identifier_nodes": self.identifier_nodes,
            "density": self.density,
            "fraud_prevalence": self.fraud_prevalence,
            "is_ring": self.is_ring,
            "cycle_count": self.cycle_count,
            "clique_count": self.clique_count,
            "reason_codes": self.reason_codes,
        }


class CommunityDetector:
    """
    Connected component analysis and clique density detection for identity graphs.
    Identifies multi-applicant synthetic identity rings, circular attribute swapping,
    and fraud contagion across relational identity subgraphs.
    """

    def __init__(
        self,
        engine: RelationalIdentityGraphEngine,
        min_ring_size: int = 2,
        ring_density_threshold: float = 0.30,
    ) -> None:
        self.engine = engine
        self.min_ring_size = min_ring_size
        self.ring_density_threshold = ring_density_threshold

    def detect_community(self, applicant_id: str, depth: int = 3) -> CommunitySummary:
        """
        Extracts the connected identity component around applicant_id (default depth=3 to capture
        all peripheral identifiers of connected members) and calculates ring metrics.
        """
        # Depth 3 reaches: root (0) -> identifiers (1) -> other applicants (2) -> their identifiers (3)
        subnetwork = self.engine.find_linked_clusters(applicant_id, depth=depth)
        G = subnetwork.graph

        cluster_id = f"cluster_{uuid.uuid5(uuid.NAMESPACE_DNS, applicant_id).hex[:12]}"

        if applicant_id not in G or G.number_of_nodes() == 0:
            return CommunitySummary(
                cluster_id=cluster_id,
                root_applicant_id=applicant_id,
                total_nodes=1,
                total_edges=0,
                applicant_members=[
                    ClusterMember(
                        applicant_id=applicant_id,
                        status="pending",
                        risk_score=0.04,
                        is_fraud=False,
                    )
                ],
                identifier_nodes=[],
                density=0.0,
                fraud_prevalence=0.0,
                is_ring=False,
                cycle_count=0,
                clique_count=0,
                reason_codes=["ISOLATED_CLEAN"],
            )

        # Get the connected component containing applicant_id
        connected_node_set = nx.node_connected_component(G, applicant_id)
        C = G.subgraph(connected_node_set)

        total_nodes = C.number_of_nodes()
        total_edges = C.number_of_edges()
        density = round((2.0 * total_edges) / (total_nodes * (total_nodes - 1)), 4) if total_nodes > 1 else 0.0

        # Partition nodes into applicants and identifiers
        applicant_nodes = [
            n for n in C.nodes
            if C.nodes[n].get("type") in ("applicant", NodeType.APPLICANT.value)
        ]
        identifier_nodes = [
            n for n in C.nodes
            if C.nodes[n].get("type") not in ("applicant", NodeType.APPLICANT.value)
        ]

        # Build members list with connected identifiers
        members: list[ClusterMember] = []
        fraud_count = 0
        for app_id in applicant_nodes:
            node_data = C.nodes[app_id]
            risk = float(node_data.get("risk", 0.0))
            is_fraud = bool(node_data.get("is_fraud", False))
            status = str(node_data.get("status", "pending"))

            if is_fraud or status in ("fraud", "rejected") or risk >= 0.85:
                fraud_count += 1

            # Find identifiers connected to this applicant
            connected_ids = [
                nbr for nbr in C.neighbors(app_id)
                if nbr in identifier_nodes
            ]

            members.append(
                ClusterMember(
                    applicant_id=app_id,
                    status=status,
                    risk_score=risk,
                    is_fraud=is_fraud,
                    connected_identifiers=connected_ids,
                )
            )

        fraud_prevalence = round(fraud_count / len(applicant_nodes), 4) if applicant_nodes else 0.0

        # Cycle detection in the bipartite cluster
        cycle_count = len(nx.cycle_basis(C))

        # Project bipartite graph onto applicants to compute applicant-to-applicant cliques
        clique_count = 0
        if len(applicant_nodes) >= 2 and len(identifier_nodes) >= 1:
            try:
                # One-mode projection: edge exists between applicants if they share >= 1 identifier
                G_proj = bipartite.projected_graph(C, applicant_nodes)
                maximal_cliques = list(nx.find_cliques(G_proj))
                # Count cliques with >= 3 mutually connected applicants
                clique_count = len([clq for clq in maximal_cliques if len(clq) >= 3])
            except Exception as e:
                logger.debug("Bipartite projection failed: %s", e)
                clique_count = 0

        # Ring classification logic
        reason_codes: list[str] = []
        is_ring = False

        if len(applicant_nodes) >= self.min_ring_size:
            # Evidence of ring coordination:
            has_cycles = cycle_count >= 1
            has_cliques = clique_count >= 1
            has_high_density = density >= self.ring_density_threshold and len(applicant_nodes) >= 3
            has_fraud_contagion = fraud_prevalence >= 0.50

            if has_cycles or has_cliques or has_high_density or has_fraud_contagion:
                is_ring = True
                reason_codes.append("HIGH_DENSITY_RING")

            if has_cycles:
                reason_codes.append("CIRCULAR_ATTRIBUTE_SWAP")
            if has_cliques:
                reason_codes.append("COORDINATED_CLIQUE")
            if fraud_prevalence > 0:
                reason_codes.append("FRAUD_CONTAGION_RISK")

            if not is_ring:
                reason_codes.append("BENIGN_COMMUNITY")
        else:
            reason_codes.append("ISOLATED_CLEAN")

        return CommunitySummary(
            cluster_id=cluster_id,
            root_applicant_id=applicant_id,
            total_nodes=total_nodes,
            total_edges=total_edges,
            applicant_members=members,
            identifier_nodes=identifier_nodes,
            density=density,
            fraud_prevalence=fraud_prevalence,
            is_ring=is_ring,
            cycle_count=cycle_count,
            clique_count=clique_count,
            reason_codes=reason_codes,
        )

    def detect_all_communities(self, db: Session, limit: int = 100) -> list[CommunitySummary]:
        """
        Batch-detects communities across the entire database for fraud analyst queues.
        Groups all applicants into distinct connected components.
        """
        applicant_ids = list(
            db.scalars(select(ApplicantNode.applicant_id).limit(limit)).all()
        )

        visited: set[str] = set()
        summaries: list[CommunitySummary] = []

        for app_id in applicant_ids:
            if app_id in visited:
                continue

            summary = self.detect_community(app_id, depth=3)
            # Mark all member applicants in this cluster as visited
            for m in summary.applicant_members:
                visited.add(m.applicant_id)

            if len(summary.applicant_members) >= 2 or summary.is_ring:
                summaries.append(summary)

        return summaries
