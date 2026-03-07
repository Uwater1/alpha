#!/usr/bin/env python3
"""
Script to run ICtest.py and grouptest.py on each alpha in 20D_top_cleaned.csv
and generate a combined report file for each alpha.
"""

import os
import sys
import io
import contextlib
from pathlib import Path

import pandas as pd
import numpy as np
import importlib

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha191.utils import (
    load_benchmark_csv,
    get_benchmark_members,
    format_alpha_name,
    parallel_load_stocks_with_alpha
)
from assessment import (
    get_clean_factor_and_forward_returns, 
    compute_performance_metrics,
    compute_stability_metrics,
    plot_quantile_returns_bar,
    plot_cumulative_returns_comparison
)


def get_alphas_from_csv(csv_path: str = "20D_top_cleaned.csv") -> list:
    """Extract alpha names from the CSV file."""
    df = pd.read_csv(csv_path)
    alphas = df['Alpha'].tolist()
    return alphas


def run_combined_test_for_alpha(alpha_name: str, benchmark: str = "zz800", ic_horizons: list = [1, 5, 10, 20, 30, 60], group_horizons: list = [20], n_jobs: int = -1) -> tuple:
    """
    Run both IC and Group tests for a single alpha, sharing the computation.
    Returns (ic_output, group_output) as strings.
    """
    print(f"Running combined tests for {alpha_name}...")
    
    output_buffer_ic = io.StringIO()
    output_buffer_group = io.StringIO()

    try:
        alpha_module = importlib.import_module(f"alpha191.{alpha_name}")
        func_name = alpha_name[:5] + "_" + alpha_name[5:]
        alpha_func = getattr(alpha_module, func_name)
    except (ImportError, AttributeError) as e:
        try:
            alpha_func = getattr(alpha_module, alpha_name)
        except (AttributeError, NameError):
            err_msg = f"ERROR: Could not load alpha {alpha_name}: {e}"
            return err_msg, err_msg

    codes = get_benchmark_members(benchmark)
    benchmark_df = load_benchmark_csv(benchmark)
    timeline = benchmark_df.index

    factor_results, price_results = parallel_load_stocks_with_alpha(
        codes, alpha_func, benchmark, n_jobs=n_jobs, show_progress=True
    )

    if not factor_results:
        err_msg = f"ERROR: No data loaded for alpha {alpha_name}"
        return err_msg, err_msg

    factor_matrix = pd.DataFrame(factor_results, dtype=np.float32).reindex(timeline)
    price_matrix = pd.DataFrame(price_results, dtype=np.float32).reindex(timeline)

    # Process all horizons at once
    all_horizons = sorted(list(set(ic_horizons + group_horizons)))

    factor_data, f_wide, q_wide = get_clean_factor_and_forward_returns(
        factor_matrix,
        price_matrix,
        periods=all_horizons,
        quantiles=10,
        max_loss=0.35,
        return_wide=True
    )

    valid_dates = factor_data.index.get_level_values('date').unique()
    f_wide = f_wide.reindex(valid_dates)
    q_wide = q_wide.reindex(valid_dates)

    full_results = compute_performance_metrics(factor_data, f_wide=f_wide, q_wide=q_wide)

    # 1. IC Test Output
    with contextlib.redirect_stdout(output_buffer_ic):
        last_date = factor_data.index.get_level_values('date').max()
        three_years_ago = last_date - pd.DateOffset(years=3)
        recent_data = factor_data[factor_data.index.get_level_values('date') >= three_years_ago]
        
        if not recent_data.empty:
            f_wide_recent = f_wide.loc[f_wide.index >= three_years_ago]
            q_wide_recent = q_wide.loc[q_wide.index >= three_years_ago]
            recent_results = compute_performance_metrics(recent_data, f_wide=f_wide_recent, q_wide=q_wide_recent)
        else:
            recent_results = None
        
        stability_results = compute_stability_metrics(factor_data, ic=full_results['ic'])
        generate_ic_report(alpha_name, benchmark, full_results, recent_results, stability_results)

    # 2. Group Test Output
    with contextlib.redirect_stdout(output_buffer_group):
        generate_group_report(alpha_name, benchmark, group_horizons, full_results)

    return output_buffer_ic.getvalue(), output_buffer_group.getvalue()


def generate_group_report(alpha_name: str, benchmark: str, horizons: list, results: dict, m_quantiles: int = 10):
    """Generate Group test report output."""
    header = f" ALPHA ASSESSMENT: {alpha_name.upper()} "
    print("\n" + "="*80)
    print(f"{header:^80}")
    print("="*80)
    print(f"Benchmark: {benchmark:<10} | Quantiles: {m_quantiles:<5} | Horizons: {str(horizons)}")
    print("-" * 80)
    
    for h in horizons:
        h_str = f"{h}D"
        if h_str not in results['q_stats']:
            continue

        q_stats = results['q_stats'][h_str]
        turnover = results['quantile_turnover']

        print(f"\n[ QUANTILE RETURNS & STATS ({h_str}) ]")
        print("-" * 85)
        print(f"{'Group':^8} | {'Mean Ret (%)':^15} | {'Std Error (%)':^15} | {'t-stat':^10} | {'p-value':^10} | {'Turnover':^10}")
        print("-" * 85)

        for q in range(1, m_quantiles + 1):
            m_ret = q_stats.loc[q, 'Mean'] * 100
            m_se = q_stats.loc[q, 'Std. Error'] * 100
            t_stat = q_stats.loc[q, 't-stat']
            p_val = q_stats.loc[q, 'p-value']
            q_turnover = turnover.get(q, np.nan)
            print(f"{q:^8} | {m_ret:15.4f}% | {m_se:15.4f}% | {t_stat:10.2f} | {p_val:10.4f} | {q_turnover:10.4f}")

        ls_row = q_stats.loc['Long-Short']
        print("-" * 85)
        print(f"{'L-S (T-B)':^8} | {ls_row['Mean']*100:14.4f}% | {ls_row['Std. Error']*100:15.4f}% | {ls_row['t-stat']:10.2f} | {ls_row['p-value']:10.4f} | {'-':^10}")
        print("-" * 85)

        print(f"\n[ L-S PORTFOLIO PERFORMANCE ({h_str}) ]")
        metrics = [
            ("Annualized Return", f"{ls_row['ann_ret']*100:.2f}%"),
            ("Annualized Vol", f"{ls_row['ann_vol']*100:.2f}%"),
            ("Sharpe Ratio", f"{ls_row['sharpe']:.2f}"),
            ("Max Drawdown", f"{ls_row['max_dd']*100:.2f}%"),
            ("Calmar Ratio", f"{ls_row['calmar']:.2f}"),
            ("Monotonicity", f"{results['mono_score'][h_str]:.4f}"),
            ("RRE (Stability)", f"{results.get('rre', np.nan):.4f}")
        ]

        for name, val in metrics:
            print(f"{name:<20}: {val:>10}")

        print("\n" + "-" * 40)

    print("\n" + "="*80)
    print(f"{'ASSESSMENT COMPLETE':^80}")
    print("="*80 + "\n")


def generate_ic_report(alpha_name: str, benchmark: str, full: dict, recent: dict, stability: dict = None):
    """Generate IC test report output."""
    print("\n" + "="*80)
    print(f" ALPHA preformance Report: {alpha_name} ({benchmark}) ".center(80, "="))
    print("="*80)
    
    pd.options.display.float_format = '{:.6f}'.format
    
    # IC Summary
    print("\n[1] Information Coefficient (IC) Summary")
    ic_summary = full['ic_summary']
    print(ic_summary.to_string())
    
    mean_ic_20d = ic_summary.loc['IC Mean', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['IC Mean'].iloc[0]
    ir_20d = ic_summary.loc['Risk-Adjusted IC (IR)', '20D'] if '20D' in ic_summary.columns else ic_summary.loc['Risk-Adjusted IC (IR)'].iloc[0]
    
    assessment = "Weak"
    if mean_ic_20d > 0.05 and ir_20d > 0.5: assessment = "Excellent"
    elif mean_ic_20d > 0.02 and ir_20d > 0.3: assessment = "Good"
    elif mean_ic_20d > 0.01: assessment = "Fair"
    print(f"\nAssessment: {assessment} (Based on 20D IC: {mean_ic_20d:.4f}, IR: {ir_20d:.4f})")
    
    # IC Decay
    print("\n[2] IC Decay Curve")
    ic_means = full['ic'].mean()
    for horizon, val in ic_means.items():
        bar = "#" * int(abs(val) * 200)
        print(f"{horizon:>5}: {val:.4f} {bar}")
    
    # Quantile Metrics
    print("\n[3] QUANTILE METRICS (SPREAD & Monotonicity)")
    mean_ret = full['mean_ret']
    spread = mean_ret.loc[10] - mean_ret.loc[1]
    mono = full['mono_score']
    
    spread_data = pd.DataFrame({
        'Q10-Q1 Spread': spread,
        'Monotonicity': mono
    }).T
    print(spread_data.to_string())
    
    # Turnover
    print("\n[4] Average Daily Turnover By Quantile")
    print(full['quantile_turnover'].to_frame().T.to_string())
    
    # Robustness
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
        
        diff = recent['ic_summary'].loc['IC Mean'].mean() / full['ic_summary'].loc['IC Mean'].mean()
        if diff > 0.9: stab = "Very Stable"
        elif diff > 0.7: stab = "Moderately Stable"
        else: stab = "Significant Decay"
        print(f"\nStability Assessment: {stab} (Recent/Full ratio: {diff:.2f})")
    
    # RRE
    print("\n[6] Rank Stability (RRE)")
    rre = full.get('rre', float('nan'))
    print(f"RRE Score: {rre:.4f} (Higher is better, 0-1 range)")
    
    pd.reset_option('display.float_format')
    print("\n" + "="*80)




def main():
    """Main function to run tests on all alphas and generate reports."""
    
    # Create report directory
    report_dir = Path("report")
    report_dir.mkdir(exist_ok=True)
    
    # Get alphas from CSV
    alphas = get_alphas_from_csv("20D_top_cleaned.csv")
    print(f"Found {len(alphas)} alphas in 20D_top_cleaned.csv")
    print(f"Alphas: {alphas}")
    
    # Open master report file once
    master_report_path = report_dir / "master_alpha_report.txt"
    with open(master_report_path, 'w') as master_f:
        master_f.write("="*80 + "\n")
        master_f.write("MASTER ALPHA TEST REPORT\n")
        master_f.write(f"Generated on: {pd.Timestamp.now()}\n")
        master_f.write("="*80 + "\n\n")

        # Process each alpha
        for i, alpha_name in enumerate(alphas, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(alphas)}] Processing {alpha_name}...")
            print(f"{'='*80}")
            
            try:
                # Run combined tests
                ic_output, group_output = run_combined_test_for_alpha(
                    alpha_name,
                    benchmark="zz800",
                    ic_horizons=[1, 5, 10, 20, 30, 60],
                    group_horizons=[20]
                )
                
                # Prepare combined report content in memory (batching)
                report_content = (
                    "="*80 + "\n"
                    f"ALPHA TEST REPORT: {alpha_name}\n"
                    "Benchmark: zz800\n"
                    "Report generated by run_alpha_tests.py\n"
                    "="*80 + "\n\n"

                    "\n" + "="*80 + "\n"
                    "IC TEST RESULTS\n"
                    "="*80 + "\n\n"
                    f"{ic_output}"

                    "\n" + "="*80 + "\n"
                    "GROUP TEST RESULTS\n"
                    "="*80 + "\n\n"
                    f"{group_output}"
                )
                
                # Write to individual report file in one go
                report_path = report_dir / f"{alpha_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report_content)

                # Append to master report
                master_f.write(report_content)
                master_f.write("\n\n" + "*"*80 + "\n\n")

                print(f"  -> Report saved to: {report_path}")

            except Exception as e:
                print(f"  -> ERROR processing {alpha_name}: {e}")
                import traceback
                traceback.print_exc()

                # Write error to report file
                error_report_path = report_dir / f"{alpha_name}_report.txt"
                with open(error_report_path, 'w') as f:
                    f.write(f"ERROR: Failed to process {alpha_name}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write(f"\nTraceback:\n{traceback.format_exc()}")
                print(f"  -> Error report saved to: {error_report_path}")
    
    print(f"\n{'='*80}")
    print(f"Complete! All reports saved to {report_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
