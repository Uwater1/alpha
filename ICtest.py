import pandas as pd
import numpy as np
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from alpha191.utils import (
    load_benchmark_csv, 
    get_benchmark_members, 
    format_alpha_name,
    parallel_load_stocks_with_alpha
)
from assessment import get_clean_factor_and_forward_returns, compute_performance_metrics, compute_stability_metrics
from datetime import datetime

def assess_alpha(alpha_name: str, benchmark: str = "zz800", horizons: List[int] = [1, 5, 10, 20, 30, 60], plot: bool = False, n_jobs: int = -1):
    """Assess an alpha using the new assessment module with multiple horizons."""
    print(f"Assessing {alpha_name} on {benchmark} with horizons {horizons} days ...")
    
    # Import alpha function
    try:
        alpha_module = importlib.import_module(f"alpha191.{alpha_name}")
        func_name = alpha_name[:5] + "_" + alpha_name[5:]
        alpha_func = getattr(alpha_module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {alpha_name}: {e}")
        try:
            alpha_func = getattr(alpha_module, alpha_name)
        except AttributeError:
            return
    
    codes = get_benchmark_members(benchmark)
    
    # Load benchmark data to get the full timeline
    benchmark_df = load_benchmark_csv(benchmark)
    timeline = benchmark_df.index
    
    # Use parallel loading for significant speedup with optimized memory
    factor_results, price_results = parallel_load_stocks_with_alpha(
        codes, alpha_func, benchmark, n_jobs=n_jobs, show_progress=True
    )

    if not factor_results:
        print("No data loaded successfully.")
        return

    # Optimize memory by using float32 and avoiding unnecessary copies
    factor_matrix = pd.DataFrame(factor_results, dtype=np.float32).reindex(timeline)
    price_matrix = pd.DataFrame(price_results, dtype=np.float32).reindex(timeline)
    
    # Use assessment module with optimized parameters
    factor_data = get_clean_factor_and_forward_returns(
        factor_matrix,
        price_matrix,
        periods=horizons,
        quantiles=10,
        max_loss=0.35  # Allow more data loss for speed
    )
    
    # Robustness Check (Last 3 years vs All)
    full_results = compute_performance_metrics(factor_data)
    
    last_date = factor_data.index.get_level_values('date').max()
    three_years_ago = last_date - pd.DateOffset(years=3)
    recent_data = factor_data[factor_data.index.get_level_values('date') >= three_years_ago]
    
    if not recent_data.empty:
        recent_results = compute_performance_metrics(recent_data)
    else:
        recent_results = None
    
    # Compute in-depth stability metrics
    stability_results = compute_stability_metrics(factor_data, ic=full_results['ic'])
        
    # Generate Enhanced Report
    generate_enhanced_report(alpha_name, benchmark, full_results, recent_results, stability_results)
    
def generate_enhanced_report(alpha_name: str, benchmark: str, full: Dict[str, Any], recent: Optional[Dict[str, Any]], stability: Optional[Dict[str, Any]] = None):
    """Prints a human-readable assessment report with in-depth stability analysis."""
    print("\n" + "="*80)
    print(f" ALPHA preformance Report: {alpha_name} ({benchmark}) ".center(80, "="))
    print("="*80)
    
    # Set pandas display options for better formatting
    pd.options.display.float_format = '{:.6f}'.format
    
    # 1. Multi-horizon IC
    print("\n[1] Information Coefficient (IC) Summary")
    ic_summary = full['ic_summary']
    print(ic_summary.to_string())
    
    # Brief assessment for IC
    mean_ic_20d = ic_summary.loc['IC Mean', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['IC Mean'].iloc[0]
    ir_20d = ic_summary.loc['Risk-Adjusted IC (IR)', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['Risk-Adjusted IC (IR)'].iloc[0]
    
    assessment = "Weak"
    if mean_ic_20d > 0.05 and ir_20d > 0.5: assessment = "Excellent"
    elif mean_ic_20d > 0.02 and ir_20d > 0.3: assessment = "Good"
    elif mean_ic_20d > 0.01: assessment = "Fair"
    print(f"\nAssessment: {assessment} (Based on 20D IC: {mean_ic_20d:.4f}, IR: {ir_20d:.4f})")
    
    # 2. IC Decay
    print("\n[2] IC Decay Curve")
    ic_means = full['ic'].mean()
    for horizon, val in ic_means.items():
        bar = "#" * int(abs(val) * 200)
        print(f"{horizon:>5}: {val:.4f} {bar}")
    
    # 3. Quantile Metrics
    print("\n[3] QUANTILE METRICS (ANALYSIS (SPREAD & Monotonicity)")
    mean_ret = full['mean_ret']
    # Spread Q10 - Q1
    spread = mean_ret.loc[10] - mean_ret.loc[1]
    mono = full['mono_score']
    
    spread_data = pd.DataFrame({
        'Q10-Q1 Spread': spread,
        'Monotonicity': mono
    }).T
    print(spread_data.to_string())
    
    # 4. Turnover by Quantile
    print("\n[4] Avergae Daily Turnover By Quantile")
    print(full['quantile_turnover'].to_frame().T.to_string())
    
    # 5. Robustness & Stability
    if recent:
        print("\n[5] Robustness: Recent 3 Years Performance")
        comparison = pd.DataFrame({
            'Metric': ['IC Mean (20D)', 'IC IR (20D)', 'IC Winrate (20D)'],
            'Full Period': [
                ic_summary.loc['IC Mean', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['IC Mean'].iloc[0],
                ic_summary.loc['Risk-Adjusted IC (IR)', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['Risk-Adjusted IC (IR)'].iloc[0],
                ic_summary.loc['IC Winrate', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['IC Winrate'].iloc[0]
            ],
            'Recent 3Y': [
                recent['ic_summary'].loc['IC Mean', '20D'] if '20D' in recent['ic_summary'].columns else recent['ic_summary'].loc['IC Mean'].iloc[0],
                recent['ic_summary'].loc['Risk-Adjusted IC (IR)', '20D'] if '20D' in recent['ic_summary'].columns else recent['ic_summary'].loc['Risk-Adjusted IC (IR)'].iloc[0],
                recent['ic_summary'].loc['IC Winrate', '20D'] if '20D' in recent['ic_summary'].columns else recent['ic_summary'].loc['IC Winrate'].iloc[0]
            ]
        })
        print(comparison.to_string(index=False))
        
        # Stability assessment
        diff = recent['ic_summary'].loc['IC Mean'].mean() / full['ic_summary'].loc['IC Mean'].mean()
        if diff > 0.9: stab = "Very Stable"
        elif diff > 0.7: stab = "Moderately Stable"
        else: stab = "Significant Decay"
        print(f"\nStability Assessment: {stab} (Recent/Full ratio: {diff:.2f})")
    
    # 6. Rank Stability (RRE)
    print("\n[6] Rank Stability (RRE)")
    rre = full.get('rre', float('nan'))
    print(f"RRE Score: {rre:.4f} (Higher is better, 0-1 range)")
    print("Description: Measures how stable the stock rankings are from day to day (Reciprocal Rank Evaluation). High RRE means low turnover.")

    # 7. In-Depth Stability Analysis (NEW)
    if stability:
        print("\n" + "="*80)
        print(" [7] IN-DEPTH STABILITY ANALYSIS ".center(80, "="))
        print("="*80)
        
        # 7a. Year-by-Year IC Breakdown
        print("\n[7a] Year-by-Year IC Performance")
        yearly = stability.get('yearly_breakdown')
        if yearly is not None and not yearly.empty:
            # Pivot for better display: Year as index, Horizon as columns
            # Show only 20D horizon for compactness, or first available
            horizon_col = '20D' if '20D' in yearly['Horizon'].values else yearly['Horizon'].iloc[0]
            yearly_pivot = yearly[yearly['Horizon'] == horizon_col].set_index('Year')[['IC Mean', 'IR', 'Winrate', 'N Days']]
            print(f"Horizon: {horizon_col}")
            print(yearly_pivot.to_string())
            
            # Show consistency: std of yearly IC means
            yearly_ic_std = yearly_pivot['IC Mean'].std()
            yearly_ic_mean = yearly_pivot['IC Mean'].mean()
            consistency = yearly_ic_mean / yearly_ic_std if yearly_ic_std > 0 else float('inf')
            print(f"\nYear-over-Year Consistency (Mean/Std): {consistency:.2f}")
        
        # 7b. IC Trend Analysis
        print("\n[7b] IC Trend Analysis")
        ic_trend = stability.get('ic_trend', {})
        if ic_trend:
            trend_data = []
            for horizon, metrics in ic_trend.items():
                trend_data.append({
                    'Horizon': horizon,
                    'Daily Slope': f"{metrics.get('slope', 0):.6f}",
                    'Annual Slope': f"{metrics.get('annual_slope', 0):.4f}",
                    'R²': f"{metrics.get('r_squared', 0):.4f}",
                    'Trend': metrics.get('interpretation', 'N/A')
                })
            trend_df = pd.DataFrame(trend_data)
            print(trend_df.to_string(index=False))
            print("\nInterpretation: Positive annual slope = improving, Negative = decaying")
        
        # 7c. Multi-Window Rolling IC
        print("\n[7c] Multi-Window Rolling IC Stats (20D horizon)")
        rolling_stats = stability.get('rolling_stats')
        if rolling_stats is not None and not rolling_stats.empty:
            # Format nicely for terminal
            formatted = rolling_stats.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            print(formatted.to_string())
            print("\nCompares recent IC across different lookback periods")
        else:
            print("Insufficient data for multi-window analysis (requires >1Y of data)")
        
        # 7d. Regime Analysis
        print("\n[7d] Regime Analysis (High IC vs Low IC Periods)")
        regime_stats = stability.get('regime_stats', {})
        if regime_stats:
            for regime_name, regime_df in regime_stats.items():
                print(f"\n{regime_name}:")
                if isinstance(regime_df, pd.DataFrame):
                    # Show transposed for compactness
                    print(regime_df.T.to_string())
        else:
            print("Regime analysis unavailable")
        
        # 7e. Overall Stability Score
        print("\n[7e] IC Consistency Score")
        print("Measures how consistent the rolling IC stays over time (higher = more stable)")
        print("-" * 60)
        stability_scores = stability.get('stability_scores')
        rolling_stats = stability.get('rolling_stats')
        if stability_scores is not None:
            for horizon, score in stability_scores.items():
                if score >= 0.8:
                    rating = "Excellent - Very consistent over time"
                elif score >= 0.5:
                    rating = "Good - Reasonably stable"
                elif score >= 0.2:
                    rating = "Fair - Some variability"
                elif score > 0:
                    rating = "Poor - High variability"
                else:
                    rating = "Unstable - IC sign changes over time"
                
                # Show score with visual bar
                print(f"\n{horizon}:")
                print(f"  Score: {score:.2f} ; Rating: {rating}")
                
                # Show min/max if available
                if rolling_stats is not None and not rolling_stats.empty:
                    if '1Y' in rolling_stats.columns:
                        try:
                            min_ic = rolling_stats.loc[('Min Rolling IC', horizon), '1Y'] if isinstance(rolling_stats.index, pd.MultiIndex) else rolling_stats.loc['Min Rolling IC', '1Y'][horizon]
                        except:
                            min_ic = rolling_stats.loc['Min Rolling IC', '1Y'] if 'Min Rolling IC' in rolling_stats.index else None
                        try:
                            max_ic = rolling_stats.loc[('Max Rolling IC', horizon), '1Y'] if isinstance(rolling_stats.index, pd.MultiIndex) else rolling_stats.loc['Max Rolling IC', '1Y'][horizon]
                        except:
                            max_ic = rolling_stats.loc['Max Rolling IC', '1Y'] if 'Max Rolling IC' in rolling_stats.index else None
                        if min_ic is not None and max_ic is not None:
                            print(f"  1Y Rolling IC Range: [{min_ic:.4f}, {max_ic:.4f}]")

    # Reset pandas display options
    pd.reset_option('display.float_format')
    print("\n" + "="*80)

def assess_alpha_legacy_print_removed():
    # This is a placeholder to show where I removed the old printing code
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assess Alpha Factor Performance")
    parser.add_argument("alpha", help="Alpha name (e.g., 1 or alpha001)")
    parser.add_argument("--horizons", type=str, default="1,5,10,20,30,60", help="Forward return horizons (default: 1,5,10,20,30,60)")
    parser.add_argument("--benchmark", default="zz800", help="Benchmark (hs300, zz500, zz800)")
    parser.add_argument("--plot", action="store_true", help="Generate tear sheet plot")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel workers (default: -1 = all CPUs)")
    
    args = parser.parse_args()
    
    alpha = format_alpha_name(args.alpha)
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    assess_alpha(alpha, args.benchmark, horizons, args.plot, args.jobs)
