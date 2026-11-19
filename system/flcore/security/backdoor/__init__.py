"""Client-side backdoor attacks for the model-exchange FL experiments.

This package intentionally stays separate from ``security.attack``: the
latter contains server-side simulations of Byzantine *update* attacks
(Min-Max, Min-Sum and Fang), while this package changes what compromised
clients train on and how their submitted model is produced.
"""

from flcore.security.backdoor.core import (BackdoorAttack, BackdoorDataset, build_backdoor_attack,
                                           resolve_backdoor_defaults)
from flcore.security.backdoor.evaluator import BackdoorEvaluator

__all__ = ["BackdoorAttack", "BackdoorDataset", "BackdoorEvaluator", "build_backdoor_attack", "resolve_backdoor_defaults"]
