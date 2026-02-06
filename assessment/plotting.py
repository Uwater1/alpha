import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats
import numpy as np

def plot_ic_ts(ic_data: pd.DataFrame, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    """
    Plots IC time series.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    if title is None:
        title = "Information Coefficient (IC) Over Time"
        
    ic_data.plot(ax=ax, title=title)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel("IC")
    return ax

def plot_quantile_returns_bar(mean_ret: pd.DataFrame, errors: Optional[pd.DataFrame] = None, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    """
    Plots mean returns by quantile as a bar chart with optional error bars.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is None:
        title = "Mean Return by Factor Quantile"
        
    mean_ret.plot(kind='bar', ax=ax, title=title, yerr=errors)
    ax.set_ylabel("Mean Return")
    ax.set_xlabel("Quantile")
    return ax

def plot_cumulative_returns_comparison(port_returns: pd.DataFrame, title: str = "Cumulative Returns: Top vs Bottom vs Long-Short", ax: Optional[plt.Axes] = None):
    """
    Plots cumulative returns for Top, Bottom, and Long-Short portfolios.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    cum_rets = (1 + port_returns).cumprod() - 1
    cum_rets.plot(ax=ax, title=title)
    
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel("Cumulative Return")
    ax.set_xlabel("Date")
    ax.legend(loc='best')
    return ax

def create_summary_tear_sheet(results: Dict[str, Any]):
    """
    Creates a summary tear sheet with IC and Quantile Return plots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    plot_ic_ts(results['ic'], ax=axes[0])
    plot_quantile_returns_bar(results['mean_ret'], ax=axes[1])
    
    plt.tight_layout()
    return fig

def plot_ic_hist(ic_data: pd.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Plots IC distribution (histogram).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    for col in ic_data.columns:
        ax.hist(ic_data[col].dropna(), bins=30, density=True, label=col, alpha=0.5)
    
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_title("IC Distribution")
    ax.legend()
    return ax

def plot_ic_qq(ic_data: pd.DataFrame, ax: Optional[plt.Axes] = None):
    """
    Plots IC Q-Q plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    for col in ic_data.columns:
        stats.probplot(ic_data[col].dropna(), dist="norm", plot=ax)
    
    ax.set_title("IC Q-Q Plot")
    return ax

def plot_cumulative_returns(returns: pd.Series, period: str, title: str, ax: Optional[plt.Axes] = None):
    """
    Plots cumulative returns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    cum_ret = (1 + returns).cumprod() - 1
    cum_ret.plot(ax=ax, title=title)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel("Cumulative Return")
    return ax

def plot_cumulative_returns_by_quantile(quantile_returns: pd.DataFrame, period: str, ax: Optional[plt.Axes] = None):
    """
    Plots cumulative returns by quantile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    cum_rets = (1 + quantile_returns).cumprod() - 1
    cum_rets.plot(ax=ax, title=f"Cumulative Returns by Quantile ({period})")
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel("Cumulative Return")
    return ax

def plot_top_bottom_quantile_turnover(quantile_turnover: pd.DataFrame, period: int, ax: Optional[plt.Axes] = None):
    """
    Plots top and bottom quantile turnover.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    max_q = quantile_turnover.columns.max()
    min_q = quantile_turnover.columns.min()
    
    quantile_turnover[[min_q, max_q]].plot(ax=ax, title=f"Top and Bottom Quantile Turnover ({period}D)")
    ax.set_ylabel("Turnover Rate")
    return ax
