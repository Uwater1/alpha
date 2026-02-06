from .utils import get_clean_factor_and_forward_returns
from .performance import (
    compute_performance_metrics,
    factor_information_coefficient,
    mean_return_by_quantile,
    factor_rank_autocorrelation
)

try:
    from .plotting import (
        create_summary_tear_sheet,
        plot_ic_ts,
        plot_quantile_returns_bar,
        plot_cumulative_returns_comparison
    )
    from .tears import create_full_tear_sheet
except ImportError as e:
    error_msg = str(e)
    def _raise_missing_dependency_error(*args, **kwargs):
        raise ImportError(
            f"Plotting functionality is unavailable because of a missing dependency: {error_msg}. "
            "Please ensure that all packages in requirements.txt (including matplotlib, seaborn, scipy) are installed. "
            "If using a virtual environment, make sure it is activated."
        )
    
    create_summary_tear_sheet = _raise_missing_dependency_error
    plot_ic_ts = _raise_missing_dependency_error
    plot_quantile_returns_bar = _raise_missing_dependency_error
    plot_cumulative_returns_comparison = _raise_missing_dependency_error
    create_full_tear_sheet = _raise_missing_dependency_error
