"""Robust aggregation defenses used before the server averages uploads."""

from flcore.security.defense.dnc import DnCDefense
from flcore.security.defense.cosine_cluster import CosineClusterDefense
from flcore.security.defense.proto_clean import ProtoCleanDefense
from flcore.security.defense.proto_probe import ProtoProbeDefense
from flcore.security.defense.pgbd_repair import (
    PGBDRepair, PGBDRepairResult, load_reference_npz,
)
from flcore.security.defense.scalar_model import ScalarModelDefense
from flcore.security.defense.robust_agr import (
    BulyanDefense, CoordinateDefense, KrumDefense, MultiKrumDefense,
)

__all__ = [
    "DnCDefense", "CosineClusterDefense", "KrumDefense", "MultiKrumDefense",
    "CoordinateDefense", "BulyanDefense", "ProtoCleanDefense", "ProtoProbeDefense",
    "PGBDRepair", "PGBDRepairResult", "load_reference_npz", "ScalarModelDefense",
]
