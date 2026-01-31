"""
Utility functions for Alpha191 factors.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_stock_csv(code: str, benchmark: str = 'zz800') -> pd.DataFrame:
    """
    Load stock data from CSV file based on benchmark selection.

    Looks for {code}.csv in the appropriate directory based on benchmark:
    - 'hs300': bao/hs300/ only
    - 'zz500': bao/zz500/ only
    - 'zz800': bao/hs300/ or bao/zz500/ (searches both)

    Parameters
    ----------
    code : str
        Stock code (e.g., 'sh_600016', 'sz_000001')
    benchmark : str, default 'zz800'
        Benchmark selection: 'hs300', 'zz500', or 'zz800'

    Returns
    -------
    pd.DataFrame
        DataFrame with date as index, sorted by date
    """
    if benchmark not in ['hs300', 'zz500', 'zz800']:
        raise ValueError(
            f"Invalid benchmark '{benchmark}'. Must be one of: 'hs300', 'zz500', 'zz800'"
        )

    if benchmark == 'hs300':
        search_paths = [Path('bao/hs300') / f'{code}.csv']
    elif benchmark == 'zz500':
        search_paths = [Path('bao/zz500') / f'{code}.csv']
    else:  # zz800
        search_paths = [
            Path('bao/hs300') / f'{code}.csv',
            Path('bao/zz500') / f'{code}.csv'
        ]

    file_path = None
    for path in search_paths:
        if path.exists():
            file_path = path
            break

    if file_path is None:
        raise FileNotFoundError(f"File '{code}.csv' not found for benchmark {benchmark}")

    df = pd.read_csv(file_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Add VWAP column if missing, approximating as amount / volume
    if 'vwap' not in df.columns:
        need_ohlc = False
        if {'amount', 'volume'}.issubset(df.columns):
            vwap_calc = df['amount'] / df['volume'].replace(0, np.nan)
            valid = (
                df['amount'].ne(0) & df['volume'].ne(0) & vwap_calc.notna() & vwap_calc.between(df['low'], df['high'])
            )
            need_ohlc = ~valid.all()
            df['vwap'] = vwap_calc
        else:
            need_ohlc = True

        if need_ohlc:
            if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
                raise ValueError(
                    "VWAP column not found and cannot be approximated (missing 'open', 'high', 'low', or 'close' columns)"
                )
            ohlc_avg = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            if 'valid' in locals():
                df['vwap'] = df['vwap'].where(valid, ohlc_avg)
            else:
                df['vwap'] = ohlc_avg
    
    return df


def run_alpha_factor(
    alpha_func,
    code: str,
    benchmark: str = 'zz800',
    end_date: str = "2026-01-23",
    lookback: int = 350
) -> float:
    """
    Generic runner for Alpha factors.
    """
    df = load_stock_csv(code, benchmark=benchmark)
    df = df.loc[:end_date]
    if len(df) < lookback:
        raise ValueError("insufficient history")
    df = df.iloc[-lookback:]
    value = alpha_func(df).iloc[-1]
    return float(value) if not np.isnan(value) else np.nan
