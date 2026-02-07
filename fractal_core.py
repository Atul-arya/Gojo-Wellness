"""
Fractal Core - The Mathematical Heart of Project Gojo (Sorcerry)

This module implements the two fundamental "rulers" for measuring
the roughness of reality:

1. HURST EXPONENT (H): Measures "memory" / long-range dependence
   - H = 0.5: Pure random walk (no memory)
   - H > 0.5: Persistent/trending (positive memory)
   - H < 0.5: Mean-reverting (negative memory)

2. HIGUCHI FRACTAL DIMENSION (D): Measures "complexity" / roughness
   - D = 1.0: Straight line (simple, "dead")
   - D ≈ 1.5: Random walk (natural complexity)
   - D = 2.0: Pure chaos (space-filling noise)

Together, these metrics form a "Universal Filter" that can distinguish
Signal from Noise across any domain: Text, Markets, Biometrics.

References:
- Hurst, H.E. (1951) "Long-term storage capacity of reservoirs"
- Higuchi, T. (1988) "Approach to an irregular time series"
- Mandelbrot, B. (1982) "The Fractal Geometry of Nature"
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Classification of signal behavior based on Hurst exponent."""
    MEAN_REVERTING = "mean_reverting"   # H < 0.5 - Choppy, anti-persistent
    RANDOM_WALK = "random_walk"          # H ≈ 0.5 - No memory, pure noise
    TRENDING = "trending"                # H > 0.5 - Persistent, has structure
    HIGHLY_PERSISTENT = "highly_persistent"  # H > 0.8 - Strong trend/bubble


class ComplexityType(Enum):
    """Classification of signal complexity based on Higuchi dimension."""
    SIMPLE = "simple"           # D < 1.2 - Too smooth, possibly artificial
    NATURAL = "natural"         # 1.2 ≤ D ≤ 1.7 - Healthy complexity
    COMPLEX = "complex"         # 1.7 < D < 1.9 - High variability
    CHAOTIC = "chaotic"         # D ≥ 1.9 - Near-random noise


@dataclass
class FractalSignature:
    """
    The complete fractal fingerprint of a signal.
    
    This is the output of the FractalAnalyzer - a pair of metrics
    that together describe the "roughness" and "memory" of any signal.
    """
    hurst: float                    # H ∈ [0, 1]
    higuchi: float                  # D ∈ [1, 2]
    signal_type: SignalType
    complexity_type: ComplexityType
    hurst_confidence: float         # R² of the H regression
    higuchi_confidence: float       # R² of the D regression
    interpretation: str             # Human-readable summary
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "hurst": round(self.hurst, 4),
            "higuchi": round(self.higuchi, 4),
            "signal_type": self.signal_type.value,
            "complexity_type": self.complexity_type.value,
            "hurst_confidence": round(self.hurst_confidence, 4),
            "higuchi_confidence": round(self.higuchi_confidence, 4),
            "interpretation": self.interpretation
        }


def hurst_exponent(
    signal: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate the Hurst Exponent using Detrended Fluctuation Analysis (DFA).
    
    DFA is more robust than R/S analysis for non-stationary time series.
    
    The Hurst exponent H measures the "memory" of a time series:
    - H = 0.5: Random walk (Brownian motion), no correlation in increments
    - H > 0.5: Trending/persistent - positive autocorrelation in increments
    - H < 0.5: Mean-reverting - negative autocorrelation in increments
    
    Algorithm (Detrended Fluctuation Analysis):
    1. Convert signal to cumulative sum of deviations from mean
    2. Divide into windows of size n
    3. Fit linear trend in each window, calculate RMS of residuals
    4. F(n) = avg RMS across windows
    5. H = slope of log(F(n)) vs log(n)
    
    Args:
        signal: 1D NumPy array of the time series
        min_window: Minimum window size (default: 10)
        max_window: Maximum window size (default: len(signal) // 4)
        
    Returns:
        (H, R^2) tuple - Hurst exponent and confidence (R-squared)
        
    Example:
        >>> import numpy as np
        >>> # White noise increments -> random walk has H ~0.5 when analyzed with DFA
        >>> increments = np.random.randn(1000)
        >>> h, confidence = hurst_exponent(increments)
        >>> print(f"H = {h:.3f} (should be ~0.5 for white noise)")
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    
    if n < 20:
        raise ValueError(f"Signal too short: {n} samples. Need at least 20.")
    
    # DFA works on the integrated signal (cumulative sum of deviations)
    mean_signal = np.mean(signal)
    integrated = np.cumsum(signal - mean_signal)
    
    if max_window is None:
        max_window = n // 4
    
    max_window = min(max_window, n // 4)
    min_window = max(min_window, 4)
    
    if max_window <= min_window:
        max_window = n // 4
        min_window = 4
    
    # Generate window sizes (logarithmically spaced)
    num_scales = min(20, max_window - min_window)
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        num=num_scales
    ).astype(int))
    
    # Filter window sizes that are valid
    window_sizes = window_sizes[window_sizes >= min_window]
    window_sizes = window_sizes[window_sizes <= max_window]
    window_sizes = window_sizes[window_sizes >= 4]  # Need at least 4 points for linear fit
    
    if len(window_sizes) < 3:
        return 0.5, 0.0
    
    fluctuations = []
    
    for win_size in window_sizes:
        num_windows = n // win_size
        if num_windows < 1:
            continue
        
        rms_values = []
        
        for i in range(num_windows):
            start = i * win_size
            end = start + win_size
            segment = integrated[start:end]
            
            # Linear detrending: fit y = ax + b
            x = np.arange(win_size)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Calculate RMS of residuals
            residuals = segment - trend
            rms = np.sqrt(np.mean(residuals ** 2))
            rms_values.append(rms)
        
        if rms_values:
            # Average fluctuation at this scale
            F_n = np.mean(rms_values)
            if F_n > 0:
                fluctuations.append((win_size, F_n))
    
    if len(fluctuations) < 3:
        return 0.5, 0.0
    
    # Linear regression on log-log scale: log(F(n)) = H * log(n) + c
    window_arr = np.array([x[0] for x in fluctuations])
    fluct_arr = np.array([x[1] for x in fluctuations])
    
    log_windows = np.log(window_arr)
    log_fluct = np.log(fluct_arr)
    
    # Linear regression
    coeffs = np.polyfit(log_windows, log_fluct, 1)
    h = coeffs[0]
    
    # Calculate R^2
    predicted = np.polyval(coeffs, log_windows)
    ss_res = np.sum((log_fluct - predicted) ** 2)
    ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
    
    # Clamp H to valid range [0, 1]
    h = float(np.clip(h, 0.0, 1.0))
    
    return h, float(max(0, r_squared))


def higuchi_fractal_dimension(
    signal: np.ndarray,
    k_max: Optional[int] = None
) -> Tuple[float, float]:
    """
    Calculate the Higuchi Fractal Dimension (HFD).
    
    The HFD measures the "complexity" or "roughness" of a time series:
    - D = 1.0: Straight line (simple, possibly artificial)
    - D ≈ 1.5: Random walk (natural complexity)
    - D = 2.0: Space-filling noise (chaotic)
    
    Algorithm (Higuchi 1988):
    1. Construct k new time series X_m^k from original X
       X_m^k = {X(m), X(m+k), X(m+2k), ...}
    2. Calculate "length" L_m(k) for each series
    3. Average over m to get L(k)
    4. D = -slope of log(L(k)) vs log(k)
    
    This is faster and more robust than box-counting for time series.
    
    Args:
        signal: 1D NumPy array of the time series
        k_max: Maximum interval (default: auto-calculated)
        
    Returns:
        (D, R²) tuple - Fractal dimension and confidence
        
    Example:
        >>> import numpy as np
        >>> line = np.linspace(0, 10, 1000)
        >>> d, conf = higuchi_fractal_dimension(line)
        >>> print(f"D = {d:.3f} (should be ~1.0 for straight line)")
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    
    if n < 10:
        raise ValueError(f"Signal too short: {n} samples. Need at least 10.")
    
    # Auto-calculate k_max if not provided
    if k_max is None:
        k_max = max(2, min(n // 4, 128))
    
    k_max = min(k_max, n // 2)
    
    if k_max < 2:
        return 1.0, 0.0  # Default to line with zero confidence
    
    lengths = []
    k_values = list(range(1, k_max + 1))
    
    for k in k_values:
        l_k = []
        
        for m in range(1, k + 1):
            # Number of elements in this sub-series
            num_elements = (n - m) // k + 1
            
            if num_elements < 2:
                continue
            
            # Calculate length of curve X_m^k
            indices = np.arange(m - 1, m - 1 + (num_elements - 1) * k + 1, k)
            
            if len(indices) < 2:
                continue
                
            # L_m(k) = (1/k) * sum(|X(m+ik) - X(m+(i-1)k)|) * (N-1) / (floor((N-m)/k) * k)
            differences = np.abs(np.diff(signal[indices]))
            normalization = (n - 1) / (k * k * (num_elements - 1))
            l_m = np.sum(differences) * normalization
            l_k.append(l_m)
        
        if l_k:
            lengths.append(np.mean(l_k))
        else:
            lengths.append(np.nan)
    
    # Remove NaN values
    valid_mask = ~np.isnan(lengths)
    k_arr = np.array(k_values)[valid_mask]
    l_arr = np.array(lengths)[valid_mask]
    
    if len(k_arr) < 3:
        return 1.0, 0.0
    
    # Linear regression on log-log scale: log(L(k)) = -D * log(k) + c
    log_k = np.log(k_arr)
    log_l = np.log(l_arr + 1e-10)
    
    coeffs = np.polyfit(log_k, log_l, 1)
    d = -coeffs[0]  # Negative slope gives dimension
    
    # Calculate R²
    predicted = np.polyval(coeffs, log_k)
    ss_res = np.sum((log_l - predicted) ** 2)
    ss_tot = np.sum((log_l - np.mean(log_l)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Clamp D to valid range [1, 2]
    d = np.clip(d, 1.0, 2.0)
    
    return float(d), float(r_squared)


class FractalAnalyzer:
    """
    Unified fractal analysis combining Hurst Exponent and Higuchi Dimension.
    
    This is the main interface for analyzing any signal (text, market, bio).
    It provides both raw metrics and human-interpretable classifications.
    
    Example:
        >>> analyzer = FractalAnalyzer()
        >>> signature = analyzer.analyze(price_series)
        >>> print(signature.interpretation)
        "Trending signal (H=0.72) with natural complexity (D=1.45)"
    """
    
    # Thresholds for classification
    H_RANDOM_LOW = 0.45
    H_RANDOM_HIGH = 0.55
    H_HIGHLY_PERSISTENT = 0.80
    
    D_SIMPLE = 1.20
    D_NATURAL_HIGH = 1.70
    D_CHAOTIC = 1.90
    
    def __init__(
        self,
        hurst_max_window: Optional[int] = None,
        higuchi_k_max: Optional[int] = None
    ):
        """
        Initialize the analyzer with optional parameters.
        
        Args:
            hurst_max_window: Maximum window for Hurst DFA calculation
            higuchi_k_max: Maximum k for Higuchi calculation
        """
        self.hurst_max_window = hurst_max_window
        self.higuchi_k_max = higuchi_k_max
    
    def classify_hurst(self, h: float) -> SignalType:
        """Classify signal type based on Hurst exponent."""
        if h >= self.H_HIGHLY_PERSISTENT:
            return SignalType.HIGHLY_PERSISTENT
        elif h > self.H_RANDOM_HIGH:
            return SignalType.TRENDING
        elif h < self.H_RANDOM_LOW:
            return SignalType.MEAN_REVERTING
        else:
            return SignalType.RANDOM_WALK
    
    def classify_higuchi(self, d: float) -> ComplexityType:
        """Classify complexity type based on Higuchi dimension."""
        if d < self.D_SIMPLE:
            return ComplexityType.SIMPLE
        elif d <= self.D_NATURAL_HIGH:
            return ComplexityType.NATURAL
        elif d < self.D_CHAOTIC:
            return ComplexityType.COMPLEX
        else:
            return ComplexityType.CHAOTIC
    
    def interpret(
        self,
        h: float,
        d: float,
        signal_type: SignalType,
        complexity_type: ComplexityType
    ) -> str:
        """Generate human-readable interpretation of the fractal signature."""
        
        type_desc = {
            SignalType.MEAN_REVERTING: "Mean-reverting signal",
            SignalType.RANDOM_WALK: "Random walk",
            SignalType.TRENDING: "Trending signal",
            SignalType.HIGHLY_PERSISTENT: "[!] Highly persistent (possible bubble/anomaly)"
        }
        
        complexity_desc = {
            ComplexityType.SIMPLE: "unnaturally simple (possible bot/failure)",
            ComplexityType.NATURAL: "natural complexity",
            ComplexityType.COMPLEX: "high complexity",
            ComplexityType.CHAOTIC: "[!] chaotic (near-random noise)"
        }
        
        return f"{type_desc[signal_type]} (H={h:.2f}) with {complexity_desc[complexity_type]} (D={d:.2f})"
    
    def analyze(self, signal: np.ndarray) -> FractalSignature:
        """
        Perform complete fractal analysis on a signal.
        
        Args:
            signal: 1D NumPy array of the time series
            
        Returns:
            FractalSignature with all metrics and interpretations
        """
        signal = np.asarray(signal, dtype=np.float64)
        
        # Calculate both metrics
        h, h_conf = hurst_exponent(signal, max_window=self.hurst_max_window)
        d, d_conf = higuchi_fractal_dimension(signal, k_max=self.higuchi_k_max)
        
        # Classify
        signal_type = self.classify_hurst(h)
        complexity_type = self.classify_higuchi(d)
        
        # Interpret
        interpretation = self.interpret(h, d, signal_type, complexity_type)
        
        return FractalSignature(
            hurst=h,
            higuchi=d,
            signal_type=signal_type,
            complexity_type=complexity_type,
            hurst_confidence=h_conf,
            higuchi_confidence=d_conf,
            interpretation=interpretation
        )
    
    def analyze_windowed(
        self,
        signal: np.ndarray,
        window_size: int,
        step_size: Optional[int] = None
    ) -> List[FractalSignature]:
        """
        Analyze a signal using sliding windows (for streaming/real-time).
        
        Args:
            signal: 1D NumPy array
            window_size: Size of each analysis window
            step_size: Step between windows (default: window_size // 2)
            
        Returns:
            List of FractalSignature for each window
        """
        signal = np.asarray(signal, dtype=np.float64)
        n = len(signal)
        
        if step_size is None:
            step_size = window_size // 2
        
        signatures = []
        start = 0
        
        while start + window_size <= n:
            window = signal[start:start + window_size]
            sig = self.analyze(window)
            signatures.append(sig)
            start += step_size
        
        return signatures
    
    def detect_regime_change(
        self,
        signatures: List[FractalSignature],
        h_threshold: float = 0.15,
        d_threshold: float = 0.20
    ) -> List[int]:
        """
        Detect points where the fractal regime changes significantly.
        
        This is useful for detecting:
        - Market crashes (sudden D drop = horizon collapse)
        - Bubble formation (H spike)
        - Bot takeover (D drops to ~1.0)
        
        Args:
            signatures: List of FractalSignatures from windowed analysis
            h_threshold: Minimum H change to trigger
            d_threshold: Minimum D change to trigger
            
        Returns:
            List of indices where regime changes occur
        """
        if len(signatures) < 2:
            return []
        
        change_points = []
        
        for i in range(1, len(signatures)):
            h_delta = abs(signatures[i].hurst - signatures[i-1].hurst)
            d_delta = abs(signatures[i].higuchi - signatures[i-1].higuchi)
            
            if h_delta >= h_threshold or d_delta >= d_threshold:
                change_points.append(i)
        
        return change_points


# =============================================================================
# Convenience functions for quick analysis
# =============================================================================

def quick_analyze(signal: np.ndarray) -> Dict:
    """
    Quick one-liner fractal analysis.
    
    Example:
        >>> result = quick_analyze(my_signal)
        >>> print(result['interpretation'])
    """
    analyzer = FractalAnalyzer()
    return analyzer.analyze(signal).to_dict()


def is_signal_natural(signal: np.ndarray, h_range=(0.4, 0.8), d_range=(1.2, 1.7)) -> bool:
    """
    Check if a signal exhibits "natural" fractal properties.
    
    Returns True if the signal looks like it came from a biological/organic
    source rather than a bot or pure noise.
    
    Args:
        signal: 1D NumPy array
        h_range: Acceptable Hurst range for "natural"
        d_range: Acceptable Higuchi range for "natural"
        
    Returns:
        True if signal appears natural, False otherwise
    """
    h, _ = hurst_exponent(signal)
    d, _ = higuchi_fractal_dimension(signal)
    
    return (h_range[0] <= h <= h_range[1]) and (d_range[0] <= d <= d_range[1])


def detect_anomaly(
    signal: np.ndarray,
    h_danger_high: float = 0.85,
    d_danger_low: float = 1.1,
    d_danger_high: float = 1.9
) -> Optional[str]:
    """
    Detect if a signal shows anomalous fractal properties.
    
    Returns a warning string if anomaly detected, None otherwise.
    
    Detectable anomalies:
    - Bubble: H > 0.85 (extreme persistence/trend following)
    - Bot/Failure: D < 1.1 (unnaturally simple)
    - Pure Noise: D > 1.9 (no structure at all)
    """
    h, _ = hurst_exponent(signal)
    d, _ = higuchi_fractal_dimension(signal)
    
    warnings = []
    
    if h > h_danger_high:
        warnings.append(f"BUBBLE: H={h:.3f} indicates extreme trend-following")
    
    if d < d_danger_low:
        warnings.append(f"BOT/FAILURE: D={d:.3f} indicates unnatural simplicity")
    
    if d > d_danger_high:
        warnings.append(f"NOISE: D={d:.3f} indicates no meaningful structure")
    
    return " | ".join(warnings) if warnings else None


# =============================================================================
# Main - Demo when run directly
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FRACTAL CORE - Project Gojo (Sorcerry)")
    print("Testing the Universal Rulers of Roughness")
    print("=" * 60)
    
    np.random.seed(42)
    n = 2000  # Use longer series for better DFA estimates
    
    # Test 1: White Noise (iid) - should give H ~0.5
    print("\n[Test 1] White Noise (iid increments)")
    white_noise = np.random.randn(n)
    analyzer = FractalAnalyzer()
    sig = analyzer.analyze(white_noise)
    print(f"  {sig.interpretation}")
    print(f"  Confidence: H={sig.hurst_confidence:.2f}, D={sig.higuchi_confidence:.2f}")
    
    # Test 2: Persistent Process (H > 0.5) - simulated with correlated noise
    print("\n[Test 2] Persistent Process (correlated increments)")
    # Generate persistent process using fractional differencing approximation
    persistent = np.zeros(n)
    persistent[0] = np.random.randn()
    for i in range(1, n):
        # Positive autocorrelation in increments
        persistent[i] = 0.7 * persistent[i-1] + np.random.randn() * 0.5
    sig = analyzer.analyze(persistent)
    print(f"  {sig.interpretation}")
    
    # Test 3: Anti-persistent Process (H < 0.5) - mean-reverting
    print("\n[Test 3] Anti-persistent Process (mean-reverting)")
    anti_persistent = np.zeros(n)
    anti_persistent[0] = np.random.randn()
    for i in range(1, n):
        # Negative autocorrelation - tends to reverse
        anti_persistent[i] = -0.5 * anti_persistent[i-1] + np.random.randn()
    sig = analyzer.analyze(anti_persistent)
    print(f"  {sig.interpretation}")
    
    # Test 4: Straight Line (should give D ~1.0)
    print("\n[Test 4] Straight Line (Simple/Dead)")
    line = np.linspace(0, 100, n)
    h, _ = hurst_exponent(line)
    d, _ = higuchi_fractal_dimension(line)
    print(f"  H = {h:.3f}, D = {d:.3f} (expect D ~1.0)")
    
    # Test 5: Pure White Noise (should give D ~2.0)
    print("\n[Test 5] Pure White Noise (Chaotic)")
    noise = np.random.randn(n)
    sig = analyzer.analyze(noise)
    print(f"  {sig.interpretation}")
    
    # Test 6: Random Walk - analyze its RETURNS (differences)
    print("\n[Test 6] Random Walk - analyzing returns")
    random_walk = np.cumsum(np.random.randn(n))
    returns = np.diff(random_walk)  # Get the increments
    sig = analyzer.analyze(returns)
    print(f"  Returns: {sig.interpretation}")
    
    # Also show what happens with the price series directly
    d_price, _ = higuchi_fractal_dimension(random_walk)
    print(f"  Price series Higuchi D = {d_price:.3f} (expect ~1.5)")
    
    # Test 7: Regime Detection
    print("\n[Test 7] Regime Change Detection")
    stable = np.random.randn(800) * 0.3
    volatile = np.random.randn(800) * 2.0
    regime_change = np.concatenate([stable, volatile])
    
    signatures = analyzer.analyze_windowed(regime_change, window_size=300, step_size=150)
    changes = analyzer.detect_regime_change(signatures, d_threshold=0.15)
    print(f"  Found {len(changes)} regime change(s) at window indices: {changes}")
    
    # Summary
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE:")
    print("  Hurst H:   <0.45 Mean-reverting | 0.45-0.55 Random | >0.55 Trending")
    print("  Higuchi D: ~1.0 Simple/Bot | 1.2-1.7 Natural | >1.9 Noise")
    print("=" * 60)
    print("All tests complete. The Universal Filter is operational.")
    print("=" * 60)
