import time
import os
from pathlib import Path
import io

def simulate_data_load():
    # Simulate heavy data loading
    time.sleep(0.05)

def simulate_compute(alpha_name):
    simulate_data_load()
    return f"Results for {alpha_name}"

def baseline_simulation(alphas):
    os.makedirs("report_baseline", exist_ok=True)
    start_time = time.time()
    for alpha_name in alphas:
        # IC Test
        ic_res = simulate_compute(alpha_name)
        ic_output = f"IC output: {ic_res}"

        # Group Test (Redundant)
        group_res = simulate_compute(alpha_name)
        group_output = f"Group output: {group_res}"

        report_path = Path("report_baseline") / f"{alpha_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(ic_output + "\n" + group_output)
    return time.time() - start_time

def optimized_simulation(alphas):
    os.makedirs("report_optimized", exist_ok=True)
    start_time = time.time()

    master_report_path = Path("report_optimized") / "master_report.txt"
    with open(master_report_path, 'w') as master_f:
        for alpha_name in alphas:
            # Shared computation
            res = simulate_compute(alpha_name)
            ic_output = f"IC output: {res}"
            group_output = f"Group output: {res}"

            report_content = ic_output + "\n" + group_output

            report_path = Path("report_optimized") / f"{alpha_name}_report.txt"
            with open(report_path, 'w') as f:
                f.write(report_content)

            master_f.write(report_content + "\n")

    return time.time() - start_time

if __name__ == "__main__":
    alphas = [f"alpha{i:03d}" for i in range(1, 21)] # 20 alphas

    b_time = baseline_simulation(alphas)
    o_time = optimized_simulation(alphas)

    print(f"Baseline Time: {b_time:.4f}s")
    print(f"Optimized Time: {o_time:.4f}s")
    print(f"Improvement: {(b_time - o_time) / b_time * 100:.2f}%")

    import shutil
    shutil.rmtree("report_baseline")
    shutil.rmtree("report_optimized")
