import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from alpha191.operators import ts_sum, ts_mean, ts_std, ts_min, ts_max, ts_rank, rolling_corr
import multiprocessing

def benchmark_operator(func, name, n_threads=4, n_tasks=100, size=100000, window=20):
    data = np.random.rand(size).astype(np.float32)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        if name == 'rolling_corr':
            data2 = np.random.rand(size).astype(np.float32)
            list(executor.map(lambda _: func(data, data2, window), range(n_tasks)))
        else:
            list(executor.map(lambda _: func(data, window), range(n_tasks)))
    end_time = time.time()

    return end_time - start_time

if __name__ == "__main__":
    operators = [
        (ts_sum, 'ts_sum'),
        (ts_mean, 'ts_mean'),
        (ts_std, 'ts_std'),
        (ts_min, 'ts_min'),
        (ts_max, 'ts_max'),
        (ts_rank, 'ts_rank'),
        (rolling_corr, 'rolling_corr'),
    ]

    n_threads = multiprocessing.cpu_count()
    print(f"Benchmarking with {n_threads} threads")

    for func, name in operators:
        duration = benchmark_operator(func, name, n_threads=n_threads)
        print(f"{name}: {duration:.4f} seconds")
