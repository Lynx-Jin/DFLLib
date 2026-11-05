"""K=2 majority selection and persistent DBSCAN groups in scalar feature space."""

import numpy as np
from sklearn.cluster import DBSCAN

from flcore.security.defense.scalar_concentration import ScalarConcentrationFilter


class ScalarDBSCANFilter(ScalarConcentrationFilter):
    """Reuse KMeans/history; replace only the spatial rectangular scan.

    Inputs are client IDs, submitted-model norms and model/global cosines.
    DBSCAN noise remains eligible for aggregation. A non-noise component is
    suspicious only if its complete two-feature diameter is sufficiently small;
    connected chains are not treated as near-identical groups.
    """

    history_policy = "stage2_eligible_clients"

    def __init__(self, eps=1.0, max_diameter=1.0, **kwargs):
        if kwargs.get("k", 2) != 2:
            raise ValueError("DBSCAN defense requires K=2 majority selection")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("DBSCAN eps must be finite and positive")
        if not np.isfinite(max_diameter) or max_diameter <= 0:
            raise ValueError("DBSCAN max_diameter must be finite and positive")
        kwargs["k"] = 2
        super().__init__(**kwargs)
        self.eps = float(eps)
        self.max_diameter = float(max_diameter)

    def _history_observed_ids(self, ids, scope_positions):
        # Stage-one exclusions are not DBSCAN non-hits: the clients were never
        # examined by stage two. Preserve both client and pair histories then.
        return [ids[int(position)] for position in scope_positions]

    def _find_groups(self, rs, cs, scope_positions, ids, width_r, width_c, threshold):
        # Construct distances directly from differences to avoid cancellation
        # when a large shared model norm is scaled by a tiny precision floor.
        distances = np.maximum(
            np.abs(rs[:, None] - rs[None, :]) / width_r,
            np.abs(cs[:, None] - cs[None, :]) / width_c,
        )
        estimator = DBSCAN(eps=self.eps, min_samples=threshold,
                           metric="precomputed").fit(distances)
        labels = estimator.labels_
        core = set(int(position) for position in estimator.core_sample_indices_)
        scope_ids = [ids[int(position)] for position in scope_positions]
        groups, diagnostics = [], []
        for label in sorted(set(int(value) for value in labels) - {-1}):
            positions = np.flatnonzero(labels == label)
            members = [scope_ids[int(position)] for position in positions]
            diameter = float(np.max(distances[np.ix_(positions, positions)]))
            compact = diameter <= self.max_diameter
            candidate = compact and len(members) >= threshold
            if candidate:
                groups.append(frozenset(members))
            diagnostics.append({
                "label": label, "ids": members, "size": len(members),
                "core_ids": [scope_ids[int(position)] for position in positions if int(position) in core],
                "norm_span": float(np.ptp(rs[positions])),
                "cosine_span": float(np.ptp(cs[positions])),
                "diameter": diameter, "compact": compact, "candidate": candidate,
                "reason": "compact" if candidate else "diameter_exceeds_limit",
            })
        max_neighbors = int(np.max(np.sum(distances <= self.eps, axis=1)))
        return groups, max_neighbors, {
            "detector": "dbscan", "dbscan_eps": self.eps,
            "dbscan_min_samples": int(threshold),
            "dbscan_max_diameter": self.max_diameter,
            "dbscan_labels": {cid: int(label) for cid, label in zip(scope_ids, labels)},
            "dbscan_clusters": diagnostics,
            "dbscan_noise_ids": [cid for cid, label in zip(scope_ids, labels) if label == -1],
            "max_neighbor_count": max_neighbors,
            "max_compact_group_size": max((len(group) for group in groups), default=0),
        }
