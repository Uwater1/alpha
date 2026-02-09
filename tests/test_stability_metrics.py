"""
Unit tests for stability metrics in assessment.performance module.
"""

import pandas as pd
import numpy as np
import pytest
from assessment.performance import (
    compute_rolling_ic_stats,
    compute_ic_trend,
    compute_yearly_ic_breakdown,
    compute_regime_analysis,
    compute_stability_metrics
)


def create_synthetic_ic(n_days=500, trend=0.0, noise=0.02, horizons=['1D', '5D', '20D']):
    """Create synthetic IC data for testing."""
    dates = pd.date_range("2020-01-01", periods=n_days, freq='B')
    data = {}
    for h in horizons:
        base_ic = 0.03 + trend * np.arange(n_days) / n_days
        ic_values = base_ic + np.random.randn(n_days) * noise
        data[h] = ic_values
    return pd.DataFrame(data, index=dates)


class TestRollingICStats:
    def test_basic_computation(self):
        """Test that rolling IC stats are computed correctly."""
        ic = create_synthetic_ic(n_days=300)
        result = compute_rolling_ic_stats(ic, windows=[252])
        
        assert not result.empty
        assert '1Y' in result.columns
        assert 'IC Mean' in result.index
        assert 'IR' in result.index
    
    def test_insufficient_data(self):
        """Test with data shorter than window."""
        ic = create_synthetic_ic(n_days=100)
        result = compute_rolling_ic_stats(ic, windows=[252])
        
        # Should return empty since data < window
        assert result.empty
    
    def test_multiple_windows(self):
        """Test with multiple window sizes."""
        ic = create_synthetic_ic(n_days=1300)
        result = compute_rolling_ic_stats(ic, windows=[252, 504, 1260])
        
        assert '1Y' in result.columns
        assert '2Y' in result.columns
        assert '5Y' in result.columns


class TestICTrend:
    def test_upward_trend(self):
        """Test detection of upward trend."""
        ic = create_synthetic_ic(n_days=500, trend=0.05, noise=0.01)
        result = compute_ic_trend(ic)
        
        for horizon in ic.columns:
            assert result[horizon]['slope'] > 0
            assert 'Improvement' in result[horizon]['interpretation']
    
    def test_downward_trend(self):
        """Test detection of downward trend."""
        ic = create_synthetic_ic(n_days=500, trend=-0.05, noise=0.01)
        result = compute_ic_trend(ic)
        
        for horizon in ic.columns:
            assert result[horizon]['slope'] < 0
            assert 'Decay' in result[horizon]['interpretation']
    
    def test_stable(self):
        """Test stable IC detection."""
        ic = create_synthetic_ic(n_days=500, trend=0.0, noise=0.01)
        result = compute_ic_trend(ic)
        
        for horizon in ic.columns:
            assert abs(result[horizon]['annual_slope']) < 0.02


class TestYearlyBreakdown:
    def test_yearly_grouping(self):
        """Test that data is grouped by year correctly."""
        ic = create_synthetic_ic(n_days=750)  # ~3 years
        result = compute_yearly_ic_breakdown(ic)
        
        assert not result.empty
        assert 'Year' in result.columns
        assert 'Horizon' in result.columns
        assert 'IC Mean' in result.columns
        
        # Should have multiple years
        years = result['Year'].unique()
        assert len(years) >= 2
    
    def test_metrics_present(self):
        """Test that all metrics are computed."""
        ic = create_synthetic_ic(n_days=300)
        result = compute_yearly_ic_breakdown(ic)
        
        assert 'IC Mean' in result.columns
        assert 'IC Std' in result.columns
        assert 'IR' in result.columns
        assert 'Winrate' in result.columns
        assert 'N Days' in result.columns


class TestRegimeAnalysis:
    def test_regime_split(self):
        """Test that regimes are correctly identified."""
        ic = create_synthetic_ic(n_days=500)
        result = compute_regime_analysis(ic)
        
        assert 'Up Market' in result or 'Down Market' in result
    
    def test_regime_stats(self):
        """Test that regime stats are computed correctly."""
        ic = create_synthetic_ic(n_days=500)
        result = compute_regime_analysis(ic)
        
        for regime_name, regime_df in result.items():
            assert 'IC Mean' in regime_df.columns
            assert 'IR' in regime_df.columns
            assert 'Winrate' in regime_df.columns


class TestComputeStabilityMetrics:
    def test_all_components_present(self):
        """Test that main function returns all components."""
        # Create minimal factor_data structure
        dates = pd.date_range("2020-01-01", periods=300, freq='B')
        tickers = ['A', 'B', 'C']
        
        data = []
        for d in dates:
            for t in tickers:
                data.append({
                    'date': d,
                    'instrument': t,
                    'factor': np.random.randn(),
                    '1D': np.random.randn() * 0.01,
                    '5D': np.random.randn() * 0.02,
                    '20D': np.random.randn() * 0.05,
                    'factor_quantile': np.random.randint(1, 11)
                })
        
        factor_data = pd.DataFrame(data).set_index(['date', 'instrument'])
        
        result = compute_stability_metrics(factor_data)
        
        assert 'rolling_stats' in result
        assert 'yearly_breakdown' in result
        assert 'ic_trend' in result
        assert 'regime_stats' in result
        assert 'stability_scores' in result
    
    def test_with_precomputed_ic(self):
        """Test that precomputed IC is used when provided."""
        dates = pd.date_range("2020-01-01", periods=300, freq='B')
        tickers = ['A', 'B', 'C']
        
        data = []
        for d in dates:
            for t in tickers:
                data.append({
                    'date': d,
                    'instrument': t,
                    'factor': np.random.randn(),
                    '1D': np.random.randn() * 0.01,
                    'factor_quantile': np.random.randint(1, 11)
                })
        
        factor_data = pd.DataFrame(data).set_index(['date', 'instrument'])
        ic = create_synthetic_ic(n_days=300, horizons=['1D'])
        
        result = compute_stability_metrics(factor_data, ic=ic)
        
        # Should work without errors
        assert 'ic_trend' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
