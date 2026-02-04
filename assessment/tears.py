import matplotlib.pyplot as plt
from . import plotting as plt_mod
from . import performance as perf
import pandas as pd
from typing import Dict, Any

def create_full_tear_sheet(factor_data: pd.DataFrame, output_path: str = "tear_sheet.png"):
    """
    Creates a full tear sheet with returns, information, and turnover analysis.
    """
    ic = perf.factor_information_coefficient(factor_data)
    mean_ret = perf.mean_return_by_quantile(factor_data)
    
    quantile_factor = factor_data['factor_quantile']
    max_q = int(quantile_factor.max())
    min_q = int(quantile_factor.min())
    
    # Calculate quantile turnover for 1D
    quantile_turnover = pd.concat([
        perf.quantile_turnover(quantile_factor, q, 1)
        for q in range(min_q, max_q + 1)
    ], axis=1)
    quantile_turnover.columns = range(min_q, max_q + 1)
    
    # 1. Plotting
    fig = plt.figure(figsize=(16, 24))
    
    # 3 columns, 4 rows
    ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2) # IC TS
    ax2 = plt.subplot2grid((5, 2), (1, 0))            # IC Hist
    ax3 = plt.subplot2grid((5, 2), (1, 1))            # IC QQ
    ax4 = plt.subplot2grid((5, 2), (2, 0), colspan=2) # Mean Ret Bar
    ax5 = plt.subplot2grid((5, 2), (3, 0), colspan=2) # Cum Ret by Quantile
    ax6 = plt.subplot2grid((5, 2), (4, 0), colspan=2) # Turnover
    
    plt_mod.plot_ic_ts(ic, ax=ax1)
    plt_mod.plot_ic_hist(ic, ax=ax2)
    plt_mod.plot_ic_qq(ic, ax=ax3)
    plt_mod.plot_quantile_returns_bar(mean_ret, ax=ax4)
    
    # For cumulative returns by quantile, we need daily returns for each quantile
    # This is a bit complex in this simplified module, but we can use the mean_quant_ret_bydate
    # if we compute it. Let's compute daily mean returns by quantile.
    return_cols = [c for c in factor_data.columns if c.endswith('D')]
    if return_cols:
        # We'll use the first return column for the cumulative plot (usually '1D' or the shortest period)
        period = return_cols[0]
        daily_quant_ret = factor_data.groupby(['date', 'factor_quantile'])[period].mean().unstack()
        plt_mod.plot_cumulative_returns_by_quantile(daily_quant_ret, period, ax=ax5)
    
    plt_mod.plot_top_bottom_quantile_turnover(quantile_turnover, 1, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Tear sheet saved to {output_path}")
    plt.close()
    return fig
