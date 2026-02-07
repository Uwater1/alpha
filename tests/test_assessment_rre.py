
import pandas as pd
import numpy as np
import pytest
from assessment.performance import factor_rre

def test_rre_perfect_stability():
    """
    Test RRE with perfectly stable ranks (should be close to 1).
    """
    # Create 3 stocks, 5 days
    # Factor values are constant, so ranks are constant
    dates = pd.date_range("2020-01-01", periods=5)
    tickers = ["A", "B", "C"]
    
    data = []
    for d in dates:
        for t, val in zip(tickers, [10, 20, 30]):
            data.append({"date": d, "instrument": t, "factor": val})
            
    df = pd.DataFrame(data).set_index(["date", "instrument"])
    
    rre_series = factor_rre(df)
    
    # RRE = 1 / (1 + KL)
    # If ranks are identical, KL should be 0, so RRE should be 1
    # We ignore the first day which is NaN
    assert np.allclose(rre_series.dropna(), 1.0, atol=1e-5)

def test_rre_max_instability():
    """
    Test RRE with maximum instability (ranks flipping).
    """
    dates = pd.date_range("2020-01-01", periods=2)
    tickers = ["A", "B"]
    
    # Day 1: A=10, B=20 (Rank A=1, B=2)
    # Day 2: A=20, B=10 (Rank A=2, B=1)
    
    data = [
        {"date": dates[0], "instrument": "A", "factor": 10},
        {"date": dates[0], "instrument": "B", "factor": 20},
        {"date": dates[1], "instrument": "A", "factor": 20},
        {"date": dates[1], "instrument": "B", "factor": 10},
    ]
    df = pd.DataFrame(data).set_index(["date", "instrument"])
    
    rre_series = factor_rre(df)
    rre_val = rre_series.dropna().iloc[0]
    
    # Expected calc:
    # Ranks Day 1: A=1, B=2. Sum=3. Probs: A=1/3, B=2/3
    # Ranks Day 2: A=2, B=1. Sum=3. Probs: A=2/3, B=1/3
    
    # KL(P_day2 || P_day1)
    # = P_A2 * log(P_A2/P_A1) + P_B2 * log(P_B2/P_B1)
    # = (2/3) * log((2/3)/(1/3)) + (1/3) * log((1/3)/(2/3))
    # = (2/3) * log(2) + (1/3) * log(0.5)
    # = 0.666 * 0.693 + 0.333 * (-0.693)
    # = 0.462 - 0.231 = 0.231
    
    # RRE = 1 / (1 + 0.231) = 1 / 1.231 = 0.812
    expected_kl = (2/3) * np.log(2) + (1/3) * np.log(0.5)
    expected_rre = 1 / (1 + expected_kl)
    
    assert np.isclose(rre_val, expected_rre, atol=1e-5)
    
    print(f"Calculated RRE: {rre_val}")
    print(f"Expected RRE: {expected_rre}")
    print(f"KL Divergence: {expected_kl}")
    
    # Calculate ranks manually to check
    # Day 1: A=10, B=20 -> Rank A=1, B=2 -> P=[1/3, 2/3]
    # Day 2: A=20, B=10 -> Rank A=2, B=1 -> P=[2/3, 1/3]
    
    assert np.isclose(rre_val, expected_rre, atol=1e-5)

if __name__ == "__main__":
    test_rre_perfect_stability()
    test_rre_max_instability()
    print("All tests passed!")
