import time
import os
from pathlib import Path

def simulate_compute(alpha_name):
    # Simulate loading data and computing alpha
    time.sleep(0.01) # Simulate some CPU work
    return f"Results for {alpha_name}"

def baseline_io_and_compute(alphas):
    os.makedirs("report_baseline", exist_ok=True)
    start_time = time.time()
    for alpha_name in alphas:
        # Simulate IC Test (Compute + Output)
        ic_output = simulate_compute(alpha_name)
        # Simulate Group Test (Redundant Compute + Output)
        group_output = simulate_compute(alpha_name)

        report_path = Path("report_baseline") / f"{alpha_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"ALPHA TEST REPORT: {alpha_name}\n")
            f.write("="*80 + "\n\n")
            f.write(ic_output + "\n")
            f.write(group_output + "\n")

    end_time = time.time()
    return end_time - start_time

def optimized_io_and_compute(alphas):
    os.makedirs("report_optimized", exist_ok=True)
    start_time = time.time()

    master_report_path = Path("report_optimized") / "master_report.txt"
    with open(master_report_path, 'w') as master_f:
        master_f.write("MASTER ALPHA TEST REPORT\n")
        master_f.write("="*80 + "\n\n")

        for alpha_name in alphas:
            # Shared computation (Done once)
            shared_results = simulate_compute(alpha_name)

            # Prepare report content in memory
            report_content = "="*80 + "\n"
            report_content += f"ALPHA TEST REPORT: {alpha_name}\n"
            report_content += "="*80 + "\n\n"
            report_content += shared_results + "\n"
            report_content += shared_results + "\n" # Reusing results

            # Write to individual report
            report_path = Path("report_optimized") / f"{alpha_name}_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_content)

            # Write to master report
            master_f.write(report_content)
            master_f.write("\n\n")

    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    alphas = [f"alpha{i:03d}" for i in range(1, 51)] # 50 alphas for testing

    print(f"Running baseline simulation with {len(alphas)} alphas...")
    baseline_time = baseline_io_and_compute(alphas)
    print(f"Baseline Time: {baseline_time:.4f}s")

    print(f"Running optimized simulation with {len(alphas)} alphas...")
    optimized_time = optimized_io_and_compute(alphas)
    print(f"Optimized Time: {optimized_time:.4f}s")

    improvement = (baseline_time - optimized_time) / baseline_time * 100
    print(f"Estimated Improvement: {improvement:.2f}%")

    # Cleanup
    import shutil
    shutil.rmtree("report_baseline")
    shutil.rmtree("report_optimized")
