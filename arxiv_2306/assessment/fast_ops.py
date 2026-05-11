import ctypes
import numpy as np
import os
from pathlib import Path

# Load the shared library
_parent = Path(__file__).parent
lib_path = _parent / "_fast_ops.so"
if not lib_path.exists():
    lib_path = _parent / "_fast_ops.dll"

try:
    _lib = ctypes.CDLL(str(lib_path))
    HAS_CPP = True
except Exception:
    HAS_CPP = False

if HAS_CPP:
    # compute_matrix_ic(const float* f_matrix, const float* r_matrix, double* results, int n_dates, int n_assets)
    _lib.compute_matrix_ic.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int
    ]
    
    # compute_quantile_means(const int8_t* q_matrix, const float* r_matrix, double* results, int n_dates, int n_assets, int n_quantiles)
    _lib.compute_quantile_means.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    # compute_quantile_turnover(const int8_t* q_matrix, double* results, int n_dates, int n_assets, int n_quantiles, int period)
    _lib.compute_quantile_turnover.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]

    # compute_rank(const float* x, float* results, int n)
    _lib.compute_rank.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]

def fast_matrix_ic(f_matrix, r_matrix):
    if not HAS_CPP:
        return None
    
    n_dates, n_assets = f_matrix.shape
    results = np.empty(n_dates, dtype=np.float64)
    
    # Ensure contiguous memory
    f_matrix = np.ascontiguousarray(f_matrix, dtype=np.float32)
    r_matrix = np.ascontiguousarray(r_matrix, dtype=np.float32)
    
    _lib.compute_matrix_ic(
        f_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        r_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_dates,
        n_assets
    )
    return results

def fast_quantile_means(q_matrix, r_matrix, n_quantiles):
    if not HAS_CPP:
        return None
    
    n_dates, n_assets = q_matrix.shape
    results = np.empty(n_quantiles, dtype=np.float64)
    
    q_matrix = np.ascontiguousarray(q_matrix, dtype=np.int8)
    r_matrix = np.ascontiguousarray(r_matrix, dtype=np.float32)
    
    _lib.compute_quantile_means(
        q_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        r_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_dates,
        n_assets,
        n_quantiles
    )
    return results

def fast_quantile_turnover(q_matrix, n_quantiles, period=1):
    # Disabled for perfect pandas consistency
    return None

def fast_rank(x):
    if not HAS_CPP:
        return None
    
    n = len(x)
    results = np.empty(n, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    
    _lib.compute_rank(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        results.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    return results
