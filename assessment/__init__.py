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
        plot_quantile_returns_bar
    )
    from .tears import create_full_tear_sheet
except ImportError:
    # Plotting might fail if matplotlib/seaborn are not properly installed
    pass
