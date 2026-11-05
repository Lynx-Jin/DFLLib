"""Detect persistent concentration using only client norm/cosine scalars."""

from collections import deque
from itertools import combinations

import numpy as np


def cosine_kmeans(values, k):
    """Deterministic one-dimensional Lloyd iterations, ordered by center."""
    values = np.asarray(values, dtype=np.float64)
    unique = np.unique(values)
    k = min(int(k), len(unique))
    centers = np.quantile(values, np.linspace(0.0, 1.0, k))
    if len(np.unique(centers)) != k:
        centers = unique[np.linspace(0, len(unique) - 1, k).round().astype(int)]
    labels = np.full(len(values), -1, dtype=int)
    for _ in range(50):
        assigned = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        for index in range(k):
            members = values[labels == index]
            if len(members):
                centers[index] = np.mean(members)
    order = np.argsort(centers, kind="stable")
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels], centers[order]


class ScalarConcentrationFilter(object):
    """MAD windows and participation-aware client/pair persistence.

    No model vectors, attack identity, malicious fraction, or ground-truth IDs
    are accepted. With K=2, keep the largest cluster before window detection.
    With K=3, scope controls window detection and other clusters are retained.
    """

    history_policy = "participating_clients"

    def __init__(self, k=2, norm_width=0.001, cosine_width=0.001,
                 min_group=3, history_window=10, min_history=3,
                 frequency=0.8, scope="all", cosine_atol=1e-6,
                 norm_rtol=1e-6, clean_max_count=1):
        if k not in (2, 3) or scope not in ("all", "main"):
            raise ValueError("k must be 2/3 and scope must be all/main")
        if min(norm_width, cosine_width, cosine_atol, norm_rtol) <= 0:
            raise ValueError("Window widths and numerical tolerances must be positive")
        if min_group < 2 or not 1 <= min_history <= history_window:
            raise ValueError("Require min_group >= 2 and 1 <= min_history <= window")
        if not 0.0 < frequency <= 1.0 or clean_max_count < 0:
            raise ValueError("Invalid frequency or clean occupancy threshold")
        self.k = int(k)
        self.norm_width = float(norm_width)
        self.cosine_width = float(cosine_width)
        self.min_group = int(min_group)
        self.window = int(history_window)
        self.min_history = int(min_history)
        self.frequency = float(frequency)
        self.scope = scope
        self.cosine_atol = float(cosine_atol)
        self.norm_rtol = float(norm_rtol)
        self.clean_max_count = int(clean_max_count)
        self.client_history = {}
        self.pair_history = {}

    @staticmethod
    def _robust_scale(values, floor):
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, max(1.4826 * mad, floor)

    def _find_groups(self, rs, cs, scope_positions, ids, width_r, width_c, threshold):
        """Original rectangular scan; subclasses can supply a density backend."""
        windows = set()
        max_count = 0
        # An optimal axis-aligned window has lower edges at observed values.
        # Search those anchors exactly rather than using grid bins.
        for left in np.unique(rs):
            in_r = (rs >= left) & (rs <= left + width_r)
            for bottom in np.unique(cs):
                members = scope_positions[in_r & (cs >= bottom) & (cs <= bottom + width_c)]
                max_count = max(max_count, len(members))
                if len(members) >= threshold:
                    windows.add(frozenset(ids[int(position)] for position in members))
        # Keep every maximal candidate window, not only the largest one.
        groups = [group for group in windows if not any(group < other for other in windows)]
        return groups, max_count, {"detector": "window"}

    def _history_observed_ids(self, ids, scope_positions):
        """Preserve the original window method's participation-based history."""
        return ids

    def apply(self, norms, cosines, round_index):
        if not norms or set(norms) != set(cosines):
            raise ValueError("Norm/cosine maps must have the same nonempty client IDs")
        ids = sorted(norms)
        r = np.asarray([norms[cid] for cid in ids], dtype=np.float64)
        c = np.asarray([cosines[cid] for cid in ids], dtype=np.float64)
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(c)):
            raise ValueError("Client statistics must be finite")
        if np.any(r < 0) or np.any(np.abs(c) > 1.0 + 1e-5):
            raise ValueError("Invalid norm or cosine statistic")

        labels, centers = cosine_kmeans(c, self.k)
        main = max(range(len(centers)), key=lambda j: (int(np.sum(labels == j)), centers[j]))
        effective_scope = "main" if self.k == 2 else self.scope
        scope_positions = np.arange(len(ids)) if effective_scope == "all" else np.flatnonzero(labels == main)
        cluster_rejected = [cid for i, cid in enumerate(ids) if self.k == 2 and labels[i] != main]
        rs, cs = r[scope_positions], c[scope_positions]
        median_r, scale_r = self._robust_scale(rs, self.norm_rtol * max(float(np.median(rs)), 1e-12))
        median_c, scale_c = self._robust_scale(cs, self.cosine_atol)
        # Equivalent to rectangles in MAD-normalized coordinates, with explicit
        # raw-space precision floors so copied float32 rows are not split.
        width_r = max(self.norm_width * scale_r, self.norm_rtol * max(median_r, 1e-12))
        width_c = max(self.cosine_width * scale_c, self.cosine_atol)
        threshold = max(self.min_group, self.clean_max_count + 1)

        groups, max_count, density_details = self._find_groups(
            rs, cs, scope_positions, ids, width_r, width_c, threshold)
        groups.sort(key=lambda group: (-len(group), tuple(sorted(group))))
        candidates = set().union(*groups) if groups else set()
        current_pairs = set()
        for group in groups:
            current_pairs.update(combinations(sorted(group), 2))

        observed_ids = self._history_observed_ids(ids, scope_positions)
        observed_set = set(observed_ids)
        scores = {}
        for cid in ids:
            history = self.client_history.setdefault(cid, deque(maxlen=self.window))
            if cid in observed_set:
                history.append(int(cid in candidates))
            scores[cid] = float(sum(history)) / len(history) if history else None
        # Absence is not a negative observation. Pair histories advance only
        # when both clients are observed by this detector in this round.
        for pair in combinations(observed_ids, 2):
            history = self.pair_history.setdefault(pair, deque(maxlen=self.window))
            history.append(int(pair in current_pairs))

        confirmed = []
        for cid in sorted(candidates):
            history = self.client_history[cid]
            if len(history) < self.min_history or scores[cid] < self.frequency:
                continue
            for peer in observed_ids:
                if peer == cid:
                    continue
                pair = tuple(sorted((cid, peer)))
                joint = self.pair_history[pair]
                if (pair in current_pairs and len(joint) >= self.min_history
                        and float(sum(joint)) / len(joint) >= self.frequency):
                    confirmed.append(cid)
                    break

        result = {
            "round": int(round_index),
            "k": len(centers),
            "requested_k": self.k,
            "scope": effective_scope,
            "requested_scope": self.scope,
            "main_cluster": int(main),
            "cluster_policy": "largest_only" if self.k == 2 else "retain_all",
            "cluster_ids": {cid: int(labels[i]) for i, cid in enumerate(ids)},
            "cluster_centers": [float(value) for value in centers],
            "cluster_sizes": [int(np.sum(labels == j)) for j in range(len(centers))],
            "main_ids": [cid for i, cid in enumerate(ids) if labels[i] == main],
            "groups": [sorted(group) for group in groups],
            "candidate_ids": sorted(candidates),
            "confirmed_ids": confirmed,
            "cluster_rejected_ids": cluster_rejected,
            "rejected_ids": sorted(set(cluster_rejected) | set(confirmed)),
            "history_scores": scores,
            "history_policy": self.history_policy,
            "history_observed_ids": list(observed_ids),
            "history_observation_counts": {cid: len(self.client_history[cid]) for cid in ids},
            "max_window_count": int(max_count),
            "required_count": threshold,
            "norm_window": width_r,
            "cosine_window": width_c,
            "norm_median": median_r,
            "cosine_median": median_c,
            "norm_scale": scale_r,
            "cosine_scale": scale_c,
        }
        result.update(density_details)
        result[density_details["detector"] + "_confirmed_ids"] = confirmed
        return result


class ScalarKMeansOnlyFilter(ScalarConcentrationFilter):
    """Stage-one ablation: identical majority KMeans, no density or history."""

    history_policy = 'not_used_kmeans_only'

    def __init__(self, **kwargs):
        if kwargs.get('k', 2) != 2:
            raise ValueError('KMeans majority-only mode requires k=2')
        super().__init__(**kwargs)

    def _find_groups(self, *args):
        return [], 0, {'detector': 'kmeans_only', 'stage2_enabled': False}

    def _history_observed_ids(self, ids, scope_positions):
        return []
