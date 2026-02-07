"""
Signal Processor - Preprocessing utilities for Project Gojo (Sorcerry)

This module provides utilities to prepare raw data for fractal analysis:
- Normalization and standardization
- Windowing and segmentation  
- Text-to-signal conversion (semantic embedding to time series)
- Data cleaning and validation

The goal is to transform any input (text, prices, bio signals) into
clean numerical time series ready for the FractalAnalyzer.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class SignalWindow:
    """A windowed segment of a signal with metadata."""
    data: np.ndarray
    start_idx: int
    end_idx: int
    timestamp: Optional[float] = None


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize a signal for consistent fractal analysis.
    
    Args:
        signal: Raw 1D NumPy array
        method: Normalization method
            - "zscore": Zero mean, unit variance (default)
            - "minmax": Scale to [0, 1] range
            - "robust": Use median and IQR (outlier resistant)
            
    Returns:
        Normalized signal as NumPy array
    """
    signal = np.asarray(signal, dtype=np.float64)
    
    if len(signal) == 0:
        return signal
    
    if method == "zscore":
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-10:
            return signal - mean
        return (signal - mean) / std
    
    elif method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        range_val = max_val - min_val
        if range_val < 1e-10:
            return np.zeros_like(signal)
        return (signal - min_val) / range_val
    
    elif method == "robust":
        median = np.median(signal)
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-10:
            return signal - median
        return (signal - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def segment_signal(
    signal: np.ndarray,
    window_size: int,
    step_size: Optional[int] = None,
    min_length: Optional[int] = None
) -> List[SignalWindow]:
    """
    Segment a signal into overlapping windows for analysis.
    
    Args:
        signal: 1D NumPy array
        window_size: Size of each window
        step_size: Step between windows (default: window_size // 2)
        min_length: Minimum acceptable window length (default: window_size // 2)
        
    Returns:
        List of SignalWindow objects
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    
    if step_size is None:
        step_size = window_size // 2
    
    if min_length is None:
        min_length = window_size // 2
    
    windows = []
    start = 0
    
    while start < n:
        end = min(start + window_size, n)
        window_data = signal[start:end]
        
        if len(window_data) >= min_length:
            windows.append(SignalWindow(
                data=window_data,
                start_idx=start,
                end_idx=end
            ))
        
        start += step_size
    
    return windows


def detrend_signal(
    signal: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    Remove trend from signal (useful for stationarity).
    
    Args:
        signal: 1D NumPy array
        method: Detrending method
            - "linear": Remove linear trend
            - "mean": Remove mean only
            - "diff": First difference (returns signal)
            
    Returns:
        Detrended signal
    """
    signal = np.asarray(signal, dtype=np.float64)
    
    if method == "linear":
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        trend = np.polyval(coeffs, x)
        return signal - trend
    
    elif method == "mean":
        return signal - np.mean(signal)
    
    elif method == "diff":
        return np.diff(signal)
    
    else:
        raise ValueError(f"Unknown detrending method: {method}")


def clean_signal(
    signal: np.ndarray,
    remove_nans: bool = True,
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0
) -> np.ndarray:
    """
    Clean a signal by handling NaNs and outliers.
    
    Args:
        signal: 1D NumPy array (may contain NaNs)
        remove_nans: Replace NaNs with interpolated values
        remove_outliers: Cap outliers at threshold
        outlier_threshold: Number of std devs for outlier detection
        
    Returns:
        Cleaned signal
    """
    signal = np.asarray(signal, dtype=np.float64).copy()
    
    if remove_nans:
        # Linear interpolation for NaN values
        nan_mask = np.isnan(signal)
        if nan_mask.any():
            x = np.arange(len(signal))
            signal[nan_mask] = np.interp(
                x[nan_mask],
                x[~nan_mask],
                signal[~nan_mask]
            )
    
    if remove_outliers:
        mean = np.mean(signal)
        std = np.std(signal)
        lower = mean - outlier_threshold * std
        upper = mean + outlier_threshold * std
        signal = np.clip(signal, lower, upper)
    
    return signal


def resample_signal(
    signal: np.ndarray,
    target_length: int,
    method: str = "linear"
) -> np.ndarray:
    """
    Resample a signal to a different length.
    
    Args:
        signal: 1D NumPy array
        target_length: Desired output length
        method: Interpolation method ("linear", "nearest")
        
    Returns:
        Resampled signal of target_length
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    
    if n == target_length:
        return signal
    
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    
    if method == "linear":
        return np.interp(x_new, x_old, signal)
    elif method == "nearest":
        indices = (x_new * (n - 1)).astype(int)
        return signal[indices]
    else:
        raise ValueError(f"Unknown resampling method: {method}")


# =============================================================================
# Text to Signal Conversion
# =============================================================================

def text_to_signal_simple(
    text: str,
    metric: str = "word_length"
) -> np.ndarray:
    """
    Convert text to a numerical signal using simple metrics.
    
    This is a lightweight method that doesn't require embeddings.
    For semantic analysis, use text_to_signal_semantic().
    
    Args:
        text: Input text string
        metric: What to measure per word/sentence
            - "word_length": Length of each word
            - "sentence_length": Number of words per sentence
            - "char_code": Sum of character codes (crude semantic proxy)
            
    Returns:
        1D NumPy array representing the text as a signal
    """
    if metric == "word_length":
        words = text.split()
        if not words:
            return np.array([0.0])
        return np.array([len(w) for w in words], dtype=np.float64)
    
    elif metric == "sentence_length":
        # Split on sentence boundaries
        import re
        sentences = re.split(r'[.!?]+', text)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return np.array([0.0])
        return np.array(lengths, dtype=np.float64)
    
    elif metric == "char_code":
        words = text.split()
        if not words:
            return np.array([0.0])
        return np.array([sum(ord(c) for c in w) for w in words], dtype=np.float64)
    
    else:
        raise ValueError(f"Unknown text metric: {metric}")


def text_to_signal_semantic(
    text: str,
    chunk_size: int = 50,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None
) -> np.ndarray:
    """
    Convert text to a signal using semantic embeddings.
    
    This creates a time series where each point represents
    the semantic "movement" between consecutive text chunks.
    
    Args:
        text: Input text string
        chunk_size: Number of words per chunk
        embed_fn: Function that takes text and returns embedding vector
                  If None, falls back to simple hash-based proxy
                  
    Returns:
        1D NumPy array of semantic distances between chunks
    """
    words = text.split()
    
    if len(words) < chunk_size * 2:
        # Not enough text for meaningful chunking
        return text_to_signal_simple(text, metric="word_length")
    
    # Create overlapping chunks
    chunks = []
    step = chunk_size // 2
    for i in range(0, len(words) - chunk_size + 1, step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    if len(chunks) < 2:
        return text_to_signal_simple(text, metric="word_length")
    
    # Get embeddings
    if embed_fn is None:
        # Fallback: use hash-based proxy (not semantic, but works)
        def simple_embed(s):
            # Create a pseudo-embedding from character statistics
            chars = np.array([ord(c) for c in s])
            if len(chars) == 0:
                return np.zeros(8)
            return np.array([
                np.mean(chars),
                np.std(chars),
                np.min(chars),
                np.max(chars),
                len(s),
                len(s.split()),
                sum(1 for c in s if c.isupper()),
                sum(1 for c in s if c in '.,!?;:')
            ])
        embed_fn = simple_embed
    
    embeddings = [embed_fn(chunk) for chunk in chunks]
    
    # Calculate distance between consecutive embeddings
    distances = []
    for i in range(1, len(embeddings)):
        dist = np.linalg.norm(embeddings[i] - embeddings[i-1])
        distances.append(dist)
    
    return np.array(distances, dtype=np.float64)


# =============================================================================
# Signal Statistics & Validation
# =============================================================================

def signal_stats(signal: np.ndarray) -> dict:
    """
    Calculate basic statistics for a signal.
    
    Useful for validation and debugging.
    """
    signal = np.asarray(signal, dtype=np.float64)
    
    return {
        "length": len(signal),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "nan_count": int(np.sum(np.isnan(signal))),
        "range": float(np.max(signal) - np.min(signal))
    }


def validate_signal(
    signal: np.ndarray,
    min_length: int = 20,
    max_nan_ratio: float = 0.1
) -> Tuple[bool, str]:
    """
    Validate that a signal is suitable for fractal analysis.
    
    Args:
        signal: 1D NumPy array
        min_length: Minimum acceptable length
        max_nan_ratio: Maximum acceptable ratio of NaN values
        
    Returns:
        (is_valid, reason) tuple
    """
    signal = np.asarray(signal, dtype=np.float64)
    
    if len(signal) < min_length:
        return False, f"Signal too short: {len(signal)} < {min_length}"
    
    nan_ratio = np.sum(np.isnan(signal)) / len(signal)
    if nan_ratio > max_nan_ratio:
        return False, f"Too many NaN values: {nan_ratio:.1%} > {max_nan_ratio:.1%}"
    
    if np.std(signal[~np.isnan(signal)]) < 1e-10:
        return False, "Signal has zero variance (constant)"
    
    return True, "Signal is valid"


# =============================================================================
# Pipeline Helper
# =============================================================================

class SignalPipeline:
    """
    Chainable preprocessing pipeline for signals.
    
    Example:
        >>> pipeline = SignalPipeline()
        >>> pipeline.add_step("clean", clean_signal, remove_nans=True)
        >>> pipeline.add_step("normalize", normalize_signal, method="zscore")
        >>> processed = pipeline.process(raw_signal)
    """
    
    def __init__(self):
        self.steps: List[Tuple[str, Callable, dict]] = []
    
    def add_step(self, name: str, func: Callable, **kwargs):
        """Add a processing step to the pipeline."""
        self.steps.append((name, func, kwargs))
        return self  # For chaining
    
    def process(self, signal: np.ndarray) -> np.ndarray:
        """Run signal through all pipeline steps."""
        result = signal
        for name, func, kwargs in self.steps:
            result = func(result, **kwargs)
        return result
    
    def describe(self) -> str:
        """Get a description of the pipeline steps."""
        return " -> ".join(name for name, _, _ in self.steps)


# =============================================================================
# Main - Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIGNAL PROCESSOR - Preprocessing Demo")
    print("=" * 60)
    
    # Demo: Process a noisy signal
    np.random.seed(42)
    raw = np.random.randn(100) * 10 + 50  # Noisy signal
    
    # Add some NaNs
    raw[10] = np.nan
    raw[50] = np.nan
    
    print("\n[1] Raw Signal Stats:")
    print(signal_stats(raw))
    
    # Clean and normalize
    clean = clean_signal(raw, remove_nans=True)
    normalized = normalize_signal(clean, method="zscore")
    
    print("\n[2] After Cleaning & Normalization:")
    print(signal_stats(normalized))
    
    # Demo: Text to signal
    text = "The quick brown fox jumps over the lazy dog. This is a test."
    text_signal = text_to_signal_simple(text, metric="word_length")
    print(f"\n[3] Text to Signal (word lengths): {text_signal}")
    
    # Demo: Pipeline
    print("\n[4] Pipeline Demo:")
    pipeline = SignalPipeline()
    pipeline.add_step("clean", clean_signal, remove_nans=True)
    pipeline.add_step("detrend", detrend_signal, method="linear")
    pipeline.add_step("normalize", normalize_signal, method="zscore")
    
    print(f"Pipeline: {pipeline.describe()}")
    processed = pipeline.process(raw)
    print(f"Processed signal stats: mean={np.mean(processed):.4f}, std={np.std(processed):.4f}")
    
    print("\n" + "=" * 60)
    print("Signal Processor ready.")
    print("=" * 60)
