"""
ARGUS M2 — Graph-Enhanced LightGBM
=====================================
Integrates relational structure from the 2-hop ego-network around each
applicant.  Synthetic identity rings create dense bipartite clusters where
different profiles share phone numbers, SSNs, emails, or device fingerprints.

Dataset origin  : Neo4j Synthetic Identity Benchmark (Nguyen & Chiu, 2026)
                  2,433 customer profiles | 433 labeled fraud instances
Graph structure : Applicant ─HAS_PHONE/EMAIL/SSN/DEVICE─► Hash Nodes
Extraction tool : NetworkX ego_graph (radius=2)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path


# ---------------------------------------------------------------------------
# Feature columns for M2 (tabular + graph topology)
# ---------------------------------------------------------------------------
M2_TABULAR_COLS: list[str] = [
    "customer_age",
    "income_lakhs",
    "account_type_risk",
    "velocity_1h",
    "velocity_24h",
]

M2_GRAPH_COLS: list[str] = [
    "shared_phone_degree",
    "shared_device_degree",
    "ego_density",
    "cycle_count",
]

M2_FEATURE_COLS: list[str] = M2_TABULAR_COLS + M2_GRAPH_COLS

M2_PARAMS: dict = {
    "objective":        "binary",
    "boosting_type":    "gbdt",
    "metric":           "binary_logloss",
    "learning_rate":    0.03,
    "num_leaves":       31,
    "max_depth":        6,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 6.0,
    "n_estimators":     150,
    "random_state":     42,
    "verbosity":        -1,
    "n_jobs":           -1,
}


# ---------------------------------------------------------------------------
# Graph Feature Extraction (NetworkX)
# ---------------------------------------------------------------------------

def extract_graph_features(G: nx.Graph, applicant_id: str) -> dict[str, float]:
    """
    Extract bounded 2-hop ego-network topological features for a single
    applicant node.

    Parameters
    ----------
    G            : nx.Graph — Bipartite identity graph.
                   Applicant nodes carry attribute ``type='applicant'``.
                   Identifier nodes carry ``type`` ∈ {'phone','device','ssn','email'}.
    applicant_id : str — Node key for the target applicant.

    Returns
    -------
    dict with keys:
        shared_phone_degree, shared_device_degree, shared_ssn_degree,
        ego_node_count, ego_edge_count, ego_density,
        max_neighbor_risk, cycle_count
    """
    _zero = {
        "shared_phone_degree":  0.0,
        "shared_device_degree": 0.0,
        "shared_ssn_degree":    0.0,
        "ego_node_count":       1.0,
        "ego_edge_count":       0.0,
        "ego_density":          0.0,
        "max_neighbor_risk":    0.0,
        "cycle_count":          0.0,
    }

    if applicant_id not in G:
        return _zero

    # --- 2-hop bounded ego-network ---
    ego = nx.ego_graph(G, applicant_id, radius=2)
    n_nodes = ego.number_of_nodes()
    n_edges = ego.number_of_edges()

    # --- shared attribute degrees via bipartite projection ---
    def _shared_applicants(attr_type: str) -> int:
        attr_nodes = [
            n for n in ego.neighbors(applicant_id)
            if ego.nodes[n].get("type") == attr_type
        ]
        peers: set[str] = set()
        for a in attr_nodes:
            peers.update(
                nb for nb in ego.neighbors(a)
                if nb != applicant_id
                and ego.nodes[nb].get("type") == "applicant"
            )
        return len(peers)

    phone_deg  = _shared_applicants("phone")
    device_deg = _shared_applicants("device")
    ssn_deg    = _shared_applicants("ssn")

    density = (
        (2.0 * n_edges) / (n_nodes * (n_nodes - 1))
        if n_nodes > 1
        else 0.0
    )

    # --- max neighbour risk (avoids target leakage by design: use historical labels) ---
    risks = [
        ego.nodes[n].get("risk", 0.0)
        for n in ego.nodes
        if n != applicant_id
    ]
    max_risk = float(max(risks)) if risks else 0.0

    # --- cycle count (4-cycles indicate circular attribute swapping in fraud rings) ---
    cycles = len(nx.cycle_basis(ego))

    return {
        "shared_phone_degree":  float(phone_deg),
        "shared_device_degree": float(device_deg),
        "shared_ssn_degree":    float(ssn_deg),
        "ego_node_count":       float(n_nodes),
        "ego_edge_count":       float(n_edges),
        "ego_density":          round(density, 4),
        "max_neighbor_risk":    max_risk,
        "cycle_count":          float(cycles),
    }


# ---------------------------------------------------------------------------
# M2 Model Wrapper
# ---------------------------------------------------------------------------

class M2GraphModel:
    """
    Graph-enhanced LightGBM that ingests pre-extracted topological features.

    The graph extraction step (``extract_graph_features``) must be run
    *before* training/inference and appended to the applicant DataFrame.
    """

    def __init__(self, params: dict | None = None):
        self.params = params or M2_PARAMS
        self._clf = lgb.LGBMClassifier(**self.params)
        self.feature_cols = M2_FEATURE_COLS

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "M2GraphModel":
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val[self.feature_cols], y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ]
        self._clf.fit(X_train[self.feature_cols], y_train, **fit_kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(X[self.feature_cols])[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        preds = self.predict_proba(X_test)
        return {
            "pr_auc":  round(average_precision_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, preds), 4),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str | Path) -> "M2GraphModel":
        instance = cls.__new__(cls)
        instance.feature_cols = M2_FEATURE_COLS
        instance.params = M2_PARAMS
        instance._clf = joblib.load(path)
        return instance
