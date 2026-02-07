"""
Market Adapter - Fractal Analysis for Financial Markets

This adapter applies the Universal Filter to financial price data,
implementing the Fractal Market Hypothesis (FMH):

CORE THEORY:
- Markets are stable when Time Horizon Diversity exists
- Day traders (short horizon) + Pension funds (long horizon) = stability
- Crashes occur when Diversity Collapses (everyone becomes short-term)

DETECTION CAPABILITIES:
1. Bubble Detection: Extremely high H (>0.8) = herding behavior
2. Crash Prediction: Sudden D drop = horizon collapse (liquidity crisis)
3. Regime Changes: Track H/D evolution to detect structural breaks

References:
- Peters, E. (1994) "Fractal Market Analysis"
- Mandelbrot, B. (2004) "The (Mis)behavior of Markets"
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal_core import (
    FractalAnalyzer,
    FractalSignature,
    hurst_exponent,
    higuchi_fractal_dimension,
    SignalType,
    ComplexityType
)


class MarketRegime(Enum):
    """Classification of market states based on fractal properties."""
    HEALTHY = "healthy"           # Natural H and D, stable liquidity
    TRENDING = "trending"         # High H, possible momentum
    CHOPPY = "choppy"             # Low H, mean-reverting
    BUBBLE = "bubble"             # Very high H, herding behavior
    CRASH_RISK = "crash_risk"     # D collapsing, liquidity crisis
    VOLATILE = "volatile"         # High D, increased uncertainty


@dataclass
class MarketAnalysis:
    """Complete fractal analysis of market data."""
    signature: FractalSignature
    regime: MarketRegime
    returns_h: float              # Hurst of returns
    returns_d: float              # Higuchi of returns  
    price_d: float                # Higuchi of price series
    volatility: float             # Standard deviation of returns
    alerts: List[str]             # Warning messages
    
    def to_dict(self) -> Dict:
        return {
            "hurst": round(self.returns_h, 4),
            "higuchi_returns": round(self.returns_d, 4),
            "higuchi_price": round(self.price_d, 4),
            "regime": self.regime.value,
            "volatility": round(self.volatility, 4),
            "alerts": self.alerts,
            "interpretation": self.signature.interpretation
        }


class MarketAdapter:
    """
    Fractal analysis adapter for financial market data.
    
    Analyzes price series using both Hurst Exponent and Higuchi Dimension
    to detect bubbles, predict crashes, and classify market regimes.
    
    Example:
        >>> adapter = MarketAdapter()
        >>> prices = np.array([100, 101, 103, 102, 105, 108, 107, ...])
        >>> analysis = adapter.analyze(prices)
        >>> print(analysis.regime)  # HEALTHY, TRENDING, BUBBLE, etc.
    """
    
    # Thresholds for market regime classification
    H_BUBBLE = 0.80           # Extreme herding
    H_TRENDING = 0.60         # Momentum present
    H_CHOPPY = 0.40           # Mean-reverting
    
    D_CRISIS_LOW = 1.20       # Horizon collapse (too simple)
    D_VOLATILE = 1.80         # High uncertainty
    
    def __init__(
        self,
        min_length: int = 50,
        returns_type: str = "log"
    ):
        """
        Initialize the market adapter.
        
        Args:
            min_length: Minimum price series length for analysis
            returns_type: "log" for log returns, "simple" for arithmetic
        """
        self.min_length = min_length
        self.returns_type = returns_type
        self.analyzer = FractalAnalyzer()
    
    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate returns from price series."""
        prices = np.asarray(prices, dtype=np.float64)
        
        if self.returns_type == "log":
            # Log returns: ln(P_t / P_{t-1})
            returns = np.diff(np.log(prices + 1e-10))
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = np.diff(prices) / (prices[:-1] + 1e-10)
        
        return returns
    
    def classify_regime(
        self,
        h: float,
        d_returns: float,
        d_price: float
    ) -> Tuple[MarketRegime, List[str]]:
        """
        Classify market regime based on fractal metrics.
        
        Returns:
            (regime, alerts) tuple
        """
        alerts = []
        
        # Check for bubble (extreme persistence)
        if h >= self.H_BUBBLE:
            alerts.append(f"[!] BUBBLE WARNING: H={h:.3f} indicates extreme herding")
            return MarketRegime.BUBBLE, alerts
        
        # Check for crash risk (horizon collapse)
        if d_price < self.D_CRISIS_LOW:
            alerts.append(f"[!] CRASH RISK: Price D={d_price:.3f} indicates horizon collapse")
            return MarketRegime.CRASH_RISK, alerts
        
        # Check for high volatility
        if d_returns >= self.D_VOLATILE:
            alerts.append(f"High volatility: Returns D={d_returns:.3f}")
            return MarketRegime.VOLATILE, alerts
        
        # Check for trending
        if h >= self.H_TRENDING:
            return MarketRegime.TRENDING, alerts
        
        # Check for choppy/mean-reverting
        if h <= self.H_CHOPPY:
            return MarketRegime.CHOPPY, alerts
        
        return MarketRegime.HEALTHY, alerts
    
    def analyze(self, prices: np.ndarray) -> MarketAnalysis:
        """
        Perform complete fractal analysis on a price series.
        
        Args:
            prices: 1D array of price data (OHLC close, or any price series)
            
        Returns:
            MarketAnalysis with all metrics and regime classification
        """
        prices = np.asarray(prices, dtype=np.float64)
        
        if len(prices) < self.min_length:
            raise ValueError(f"Price series too short: {len(prices)} < {self.min_length}")
        
        # Calculate returns
        returns = self._calculate_returns(prices)
        
        # Analyze returns (for Hurst - this is what matters for memory)
        h, h_conf = hurst_exponent(returns)
        d_returns, d_ret_conf = higuchi_fractal_dimension(returns)
        
        # Analyze price series (for overall complexity/structure)
        d_price, d_price_conf = higuchi_fractal_dimension(prices)
        
        # Get full signature
        signature = self.analyzer.analyze(returns)
        
        # Calculate volatility
        volatility = np.std(returns)
        
        # Classify regime
        regime, alerts = self.classify_regime(h, d_returns, d_price)
        
        return MarketAnalysis(
            signature=signature,
            regime=regime,
            returns_h=h,
            returns_d=d_returns,
            price_d=d_price,
            volatility=volatility,
            alerts=alerts
        )
    
    def analyze_rolling(
        self,
        prices: np.ndarray,
        window_size: int = 100,
        step_size: int = 20
    ) -> List[MarketAnalysis]:
        """
        Rolling window analysis for tracking regime changes over time.
        
        Args:
            prices: Full price series
            window_size: Size of rolling window
            step_size: Step between windows
            
        Returns:
            List of MarketAnalysis for each window
        """
        prices = np.asarray(prices, dtype=np.float64)
        n = len(prices)
        
        analyses = []
        start = 0
        
        while start + window_size <= n:
            window = prices[start:start + window_size]
            try:
                analysis = self.analyze(window)
                analyses.append(analysis)
            except ValueError:
                pass  # Skip windows that are too short
            start += step_size
        
        return analyses
    
    def detect_regime_changes(
        self,
        analyses: List[MarketAnalysis],
        h_threshold: float = 0.15,
        d_threshold: float = 0.20
    ) -> List[Tuple[int, str]]:
        """
        Detect significant regime changes from rolling analysis.
        
        Returns:
            List of (index, description) tuples for each detected change
        """
        if len(analyses) < 2:
            return []
        
        changes = []
        
        for i in range(1, len(analyses)):
            prev = analyses[i - 1]
            curr = analyses[i]
            
            h_delta = curr.returns_h - prev.returns_h
            d_delta = curr.price_d - prev.price_d
            
            descriptions = []
            
            # Check for H changes
            if abs(h_delta) >= h_threshold:
                direction = "increasing" if h_delta > 0 else "decreasing"
                descriptions.append(f"H {direction} ({prev.returns_h:.2f} -> {curr.returns_h:.2f})")
            
            # Check for D changes (especially drops - crash risk)
            if abs(d_delta) >= d_threshold:
                if d_delta < 0:
                    descriptions.append(f"D DROPPING ({prev.price_d:.2f} -> {curr.price_d:.2f}) - CRASH RISK")
                else:
                    descriptions.append(f"D increasing ({prev.price_d:.2f} -> {curr.price_d:.2f})")
            
            # Check for regime transitions
            if prev.regime != curr.regime:
                descriptions.append(f"Regime: {prev.regime.value} -> {curr.regime.value}")
            
            if descriptions:
                changes.append((i, " | ".join(descriptions)))
        
        return changes
    
    def get_trading_signal(self, analysis: MarketAnalysis) -> Dict:
        """
        Generate a simple trading signal based on fractal analysis.
        
        This is a basic example - real trading systems would be more sophisticated.
        
        Returns:
            Dictionary with signal and reasoning
        """
        h = analysis.returns_h
        regime = analysis.regime
        
        if regime == MarketRegime.BUBBLE:
            return {
                "signal": "REDUCE_EXPOSURE",
                "confidence": "HIGH",
                "reason": f"Bubble detected (H={h:.2f}). Extreme herding suggests reversal risk."
            }
        
        if regime == MarketRegime.CRASH_RISK:
            return {
                "signal": "EXIT",
                "confidence": "HIGH", 
                "reason": f"Horizon collapse detected. Liquidity crisis likely."
            }
        
        if regime == MarketRegime.TRENDING:
            return {
                "signal": "FOLLOW_TREND",
                "confidence": "MEDIUM",
                "reason": f"Trending market (H={h:.2f}). Momentum strategies may work."
            }
        
        if regime == MarketRegime.CHOPPY:
            return {
                "signal": "MEAN_REVERSION",
                "confidence": "MEDIUM",
                "reason": f"Choppy market (H={h:.2f}). Mean-reversion strategies may work."
            }
        
        return {
            "signal": "NEUTRAL",
            "confidence": "LOW",
            "reason": "Market in normal state. No clear edge."
        }


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MARKET ADAPTER - Fractal Market Hypothesis Demo")
    print("=" * 60)
    
    np.random.seed(42)
    n = 500
    
    # Simulate different market conditions
    adapter = MarketAdapter()
    
    # 1. Normal market (random walk with drift)
    print("\n[1] Normal Market (Random Walk with Drift)")
    normal_returns = np.random.randn(n) * 0.02 + 0.0003  # 2% vol, slight positive drift
    normal_prices = 100 * np.exp(np.cumsum(normal_returns))
    
    analysis = adapter.analyze(normal_prices)
    print(f"  Regime: {analysis.regime.value}")
    print(f"  H={analysis.returns_h:.3f}, D_returns={analysis.returns_d:.3f}, D_price={analysis.price_d:.3f}")
    print(f"  Signal: {adapter.get_trading_signal(analysis)['signal']}")
    
    # 2. Trending market (momentum)
    print("\n[2] Trending Market (Strong Momentum)")
    trending_returns = np.zeros(n)
    trending_returns[0] = np.random.randn() * 0.02
    for i in range(1, n):
        # Positive autocorrelation
        trending_returns[i] = 0.6 * trending_returns[i-1] + np.random.randn() * 0.01
    trending_prices = 100 * np.exp(np.cumsum(trending_returns))
    
    analysis = adapter.analyze(trending_prices)
    print(f"  Regime: {analysis.regime.value}")
    print(f"  H={analysis.returns_h:.3f}")
    print(f"  Signal: {adapter.get_trading_signal(analysis)['signal']}")
    
    # 3. Bubble simulation (extreme persistence)
    print("\n[3] Bubble Market (Extreme Herding)")
    bubble_returns = np.zeros(n)
    bubble_returns[0] = 0.01
    for i in range(1, n):
        # Very high autocorrelation - herding
        bubble_returns[i] = 0.85 * bubble_returns[i-1] + np.random.randn() * 0.005 + 0.002
    bubble_prices = 100 * np.exp(np.cumsum(bubble_returns))
    
    analysis = adapter.analyze(bubble_prices)
    print(f"  Regime: {analysis.regime.value}")
    print(f"  H={analysis.returns_h:.3f}")
    print(f"  Alerts: {analysis.alerts}")
    print(f"  Signal: {adapter.get_trading_signal(analysis)['signal']}")
    
    # 4. Rolling analysis with regime change
    print("\n[4] Regime Change Detection")
    # First half: normal, second half: volatile
    half = n // 2
    normal_part = np.random.randn(half) * 0.01
    volatile_part = np.random.randn(half) * 0.05
    combined_returns = np.concatenate([normal_part, volatile_part])
    combined_prices = 100 * np.exp(np.cumsum(combined_returns))
    
    analyses = adapter.analyze_rolling(combined_prices, window_size=100, step_size=50)
    changes = adapter.detect_regime_changes(analyses, d_threshold=0.15)
    
    print(f"  Analyzed {len(analyses)} windows")
    print(f"  Detected {len(changes)} regime change(s):")
    for idx, desc in changes:
        print(f"    Window {idx}: {desc}")
    
    print("\n" + "=" * 60)
    print("Market Adapter operational. Ready for real data.")
    print("=" * 60)
