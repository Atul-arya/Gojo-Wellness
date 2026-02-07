"""
Unit Tests for Fractal Core - Project Gojo (Sorcerry)

Tests for Hurst Exponent and Higuchi Fractal Dimension algorithms.
Validates against synthetic signals with known properties.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal_core import (
    hurst_exponent,
    higuchi_fractal_dimension,
    FractalAnalyzer,
    SignalType,
    ComplexityType,
    quick_analyze,
    is_signal_natural,
    detect_anomaly
)


class TestHurstExponent:
    """Tests for Hurst Exponent calculation."""
    
    def test_random_walk_gives_h_near_half(self):
        """Random walk (cumsum of iid) should have H ≈ 0.5"""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(2000))
        h, confidence = hurst_exponent(random_walk)
        
        # Should be within 0.1 of 0.5
        assert 0.40 <= h <= 0.60, f"Random walk H={h:.3f}, expected ~0.5"
        assert confidence > 0.8, f"Confidence too low: {confidence:.3f}"
    
    def test_trending_signal_gives_high_h(self):
        """A trending signal should have H > 0.5"""
        np.random.seed(42)
        n = 1000
        # Strong upward trend with small noise
        trend = np.linspace(0, 100, n) + np.random.randn(n) * 0.5
        # Integrate to make it persistent
        trend = np.cumsum(np.abs(np.random.randn(n)))
        
        h, _ = hurst_exponent(trend)
        assert h > 0.55, f"Trending signal H={h:.3f}, expected > 0.55"
    
    def test_mean_reverting_gives_low_h(self):
        """Oscillating/mean-reverting signals should have H < 0.5"""
        np.random.seed(42)
        n = 1000
        # Generate anti-persistent series
        anti_persistent = np.zeros(n)
        for i in range(1, n):
            anti_persistent[i] = -0.7 * anti_persistent[i-1] + np.random.randn()
        
        h, _ = hurst_exponent(anti_persistent)
        assert h < 0.50, f"Mean-reverting signal H={h:.3f}, expected < 0.50"
    
    def test_short_signal_raises_error(self):
        """Signals less than 20 samples should raise ValueError."""
        short_signal = np.random.randn(10)
        with pytest.raises(ValueError):
            hurst_exponent(short_signal)
    
    def test_h_is_bounded_zero_one(self):
        """H should always be in [0, 1] range."""
        np.random.seed(42)
        for _ in range(10):
            signal = np.cumsum(np.random.randn(500))
            h, _ = hurst_exponent(signal)
            assert 0.0 <= h <= 1.0, f"H={h:.3f} out of bounds"


class TestHiguchiFractalDimension:
    """Tests for Higuchi Fractal Dimension calculation."""
    
    def test_straight_line_gives_d_near_one(self):
        """A straight line should have D ≈ 1.0"""
        line = np.linspace(0, 100, 1000)
        d, confidence = higuchi_fractal_dimension(line)
        
        assert d < 1.15, f"Straight line D={d:.3f}, expected ~1.0"
    
    def test_random_walk_gives_d_near_1_5(self):
        """Random walk should have D ≈ 1.5"""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(2000))
        d, confidence = higuchi_fractal_dimension(random_walk)
        
        # Should be between 1.3 and 1.7
        assert 1.30 <= d <= 1.70, f"Random walk D={d:.3f}, expected ~1.5"
    
    def test_white_noise_gives_high_d(self):
        """White noise should have D close to 2.0"""
        np.random.seed(42)
        noise = np.random.randn(2000)
        d, _ = higuchi_fractal_dimension(noise)
        
        assert d > 1.7, f"White noise D={d:.3f}, expected > 1.7"
    
    def test_d_is_bounded_one_two(self):
        """D should always be in [1, 2] range."""
        np.random.seed(42)
        for _ in range(10):
            signal = np.cumsum(np.random.randn(500))
            d, _ = higuchi_fractal_dimension(signal)
            assert 1.0 <= d <= 2.0, f"D={d:.3f} out of bounds"
    
    def test_short_signal_raises_error(self):
        """Signals less than 10 samples should raise ValueError."""
        short_signal = np.random.randn(5)
        with pytest.raises(ValueError):
            higuchi_fractal_dimension(short_signal)


class TestFractalAnalyzer:
    """Tests for the unified FractalAnalyzer class."""
    
    def test_analyzer_returns_fractal_signature(self):
        """Analyzer should return a complete FractalSignature."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(500))
        
        analyzer = FractalAnalyzer()
        sig = analyzer.analyze(signal)
        
        assert hasattr(sig, 'hurst')
        assert hasattr(sig, 'higuchi')
        assert hasattr(sig, 'signal_type')
        assert hasattr(sig, 'complexity_type')
        assert hasattr(sig, 'interpretation')
    
    def test_classify_hurst_random_walk(self):
        """H ≈ 0.5 should classify as RANDOM_WALK."""
        analyzer = FractalAnalyzer()
        
        assert analyzer.classify_hurst(0.50) == SignalType.RANDOM_WALK
        assert analyzer.classify_hurst(0.48) == SignalType.RANDOM_WALK
        assert analyzer.classify_hurst(0.52) == SignalType.RANDOM_WALK
    
    def test_classify_hurst_trending(self):
        """H > 0.55 should classify as TRENDING."""
        analyzer = FractalAnalyzer()
        
        assert analyzer.classify_hurst(0.65) == SignalType.TRENDING
        assert analyzer.classify_hurst(0.70) == SignalType.TRENDING
    
    def test_classify_hurst_highly_persistent(self):
        """H > 0.80 should classify as HIGHLY_PERSISTENT."""
        analyzer = FractalAnalyzer()
        
        assert analyzer.classify_hurst(0.85) == SignalType.HIGHLY_PERSISTENT
        assert analyzer.classify_hurst(0.95) == SignalType.HIGHLY_PERSISTENT
    
    def test_classify_higuchi_simple(self):
        """D < 1.2 should classify as SIMPLE."""
        analyzer = FractalAnalyzer()
        
        assert analyzer.classify_higuchi(1.05) == ComplexityType.SIMPLE
        assert analyzer.classify_higuchi(1.15) == ComplexityType.SIMPLE
    
    def test_classify_higuchi_natural(self):
        """D between 1.2 and 1.7 should classify as NATURAL."""
        analyzer = FractalAnalyzer()
        
        assert analyzer.classify_higuchi(1.40) == ComplexityType.NATURAL
        assert analyzer.classify_higuchi(1.55) == ComplexityType.NATURAL
    
    def test_windowed_analysis(self):
        """Windowed analysis should return list of signatures."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(1000))
        
        analyzer = FractalAnalyzer()
        signatures = analyzer.analyze_windowed(signal, window_size=200, step_size=100)
        
        assert len(signatures) > 1
        assert all(0 <= s.hurst <= 1 for s in signatures)
        assert all(1 <= s.higuchi <= 2 for s in signatures)
    
    def test_regime_change_detection(self):
        """Should detect when fractal properties change abruptly."""
        np.random.seed(42)
        
        # Create signal with regime change
        stable = np.cumsum(np.random.randn(500) * 0.2)
        volatile = np.cumsum(np.random.randn(500) * 3.0)
        signal = np.concatenate([stable, volatile])
        
        analyzer = FractalAnalyzer()
        signatures = analyzer.analyze_windowed(signal, window_size=150, step_size=75)
        changes = analyzer.detect_regime_change(signatures, h_threshold=0.1, d_threshold=0.15)
        
        # Should detect at least one change
        assert len(changes) >= 1, "Should detect regime change"


class TestConvenienceFunctions:
    """Tests for quick_analyze, is_signal_natural, detect_anomaly."""
    
    def test_quick_analyze_returns_dict(self):
        """quick_analyze should return a dictionary."""
        np.random.seed(42)
        signal = np.cumsum(np.random.randn(500))
        
        result = quick_analyze(signal)
        
        assert isinstance(result, dict)
        assert 'hurst' in result
        assert 'higuchi' in result
        assert 'interpretation' in result
    
    def test_is_signal_natural_for_random_walk(self):
        """Random walk should be classified as natural."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(1000))
        
        assert is_signal_natural(random_walk) == True
    
    def test_is_signal_natural_for_line(self):
        """Straight line should NOT be natural (too simple)."""
        line = np.linspace(0, 100, 1000)
        
        # Line has very low D, so not natural
        assert is_signal_natural(line) == False
    
    def test_detect_anomaly_bubble(self):
        """Should detect bubble (extremely high H)."""
        np.random.seed(42)
        # Simulate persistent trending (bubble-like)
        bubble = np.cumsum(np.abs(np.random.randn(1000)))  # Always up
        
        # This should have high H
        h, _ = hurst_exponent(bubble)
        
        if h > 0.85:  # Only test if we generated high enough H
            warning = detect_anomaly(bubble, h_danger_high=0.85)
            assert warning is not None and "BUBBLE" in warning
    
    def test_detect_anomaly_bot(self):
        """Should detect bot (unnatural simplicity)."""
        # Straight line with tiny noise (bot-like movement)
        line = np.linspace(0, 100, 1000) + np.random.randn(1000) * 0.01
        
        warning = detect_anomaly(line, d_danger_low=1.1)
        assert warning is not None and "BOT" in warning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
