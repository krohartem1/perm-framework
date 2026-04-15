from .framework import (
    ExperimentConfig,
    GroupLevelMetric,
    MetricBase,
    PrecomputedMetric,
    plot_bootstrap_results,
    plot_permutation_results,
    plot_user_distributions,
    run_bootstrap_test,
    run_permutation_test,
)

from .metrics import BENMetric, BENWeightedMetric, UniqueItemsMetric, UserUniqueItemsMetric
from .utils import sample_users_by_group
try:
    from .validation_checks import run_all_checks
except Exception:  # optional dependency (scipy)
    run_all_checks = None  # type: ignore

try:
    from .linearization import compute_linearized_ben, linearized_ttest
except Exception:  # optional dependency (scipy)
    compute_linearized_ben = None  # type: ignore
    linearized_ttest = None  # type: ignore

__all__ = [
    "ExperimentConfig",
    "GroupLevelMetric",
    "MetricBase",
    "PrecomputedMetric",
    "run_permutation_test",
    "run_bootstrap_test",
    "plot_permutation_results",
    "plot_bootstrap_results",
    "plot_user_distributions",
    "BENMetric",
    "BENWeightedMetric",
    "UniqueItemsMetric",
    "UserUniqueItemsMetric",
    "sample_users_by_group",
    "run_all_checks",
    "compute_linearized_ben",
    "linearized_ttest",
]

