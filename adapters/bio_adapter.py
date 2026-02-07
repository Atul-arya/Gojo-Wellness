"""
Bio Adapter - Fractal Analysis for Mental Health Biometrics

Analyzes human biometric signals to detect emotional and mental states
through the "roughness" of their movements and voice.

THEORY:
- Human behavior has characteristic fractal signatures
- Stress/anxiety increases chaos (higher D)
- Depression reduces complexity (lower D, more predictable)
- Healthy states show "natural" complexity (D ~ 1.4-1.6)

SIGNALS ANALYZED:
1. Touch patterns: pressure, velocity, jitter
2. Voice: pitch variation, tremor, energy
3. Mouse/trackpad: movement chaos, hesitation, smoothness

APPLICATIONS:
- Mental health monitoring (non-invasive)
- Stress detection during interactions
- Early warning for emotional distress
- Wellness tracking over time
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractal_core import (
    FractalAnalyzer,
    FractalSignature,
    hurst_exponent,
    higuchi_fractal_dimension,
    SignalType,
    ComplexityType
)


class EmotionalState(Enum):
    """Emotional state classification based on fractal patterns."""
    CALM = "calm"                   # Natural H and D, stable
    FOCUSED = "focused"             # High H, natural D - in the zone
    ANXIOUS = "anxious"             # Low H, high D - jittery, chaotic
    STRESSED = "stressed"           # High D, erratic patterns
    FATIGUED = "fatigued"           # Low D, sluggish, overly smooth
    DEPRESSED = "depressed"         # Very low D, reduced variability
    AGITATED = "agitated"           # Very high D, emotional turbulence
    UNKNOWN = "unknown"


@dataclass
class MentalStateAnalysis:
    """Complete mental state analysis from biometric data."""
    signature: FractalSignature
    state: EmotionalState
    stress_level: float          # 0-1 scale
    energy_level: float          # 0-1 scale
    stability_score: float       # 0-1 scale
    alerts: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "state": self.state.value,
            "stress_level": round(self.stress_level, 3),
            "energy_level": round(self.energy_level, 3),
            "stability_score": round(self.stability_score, 3),
            "hurst": round(self.signature.hurst, 4),
            "complexity": round(self.signature.higuchi, 4),
            "alerts": self.alerts
        }


class BioAdapter:
    """
    Fractal analysis adapter for biometric/behavioral data.
    
    Analyzes touch, voice, and mouse patterns to detect mental states.
    
    Example:
        >>> adapter = BioAdapter()
        >>> # Analyze mouse movement
        >>> analysis = adapter.analyze_movement(x_coords, y_coords, timestamps)
        >>> print(f"State: {analysis.state.value}, Stress: {analysis.stress_level:.0%}")
    """
    
    # Thresholds for emotional state classification
    # Based on natural human behavioral patterns
    
    # Hurst thresholds
    H_FOCUSED = 0.65          # High persistence = focused
    H_ANXIOUS = 0.40          # Low persistence = scattered mind
    
    # Higuchi thresholds
    D_CALM_LOW = 1.35         # Below = too simple (fatigued/depressed)
    D_CALM_HIGH = 1.65        # Above = too chaotic (anxious/agitated)
    D_DEPRESSED = 1.20        # Very low = concerning
    D_AGITATED = 1.85         # Very high = emotional turbulence
    
    def __init__(self):
        """Initialize the bio adapter."""
        self.analyzer = FractalAnalyzer()
    
    # =========================================================================
    # Core Classification
    # =========================================================================
    
    def classify_state(
        self,
        h: float,
        d: float
    ) -> Tuple[EmotionalState, float, float, float, List[str]]:
        """
        Classify emotional state based on fractal metrics.
        
        Returns:
            (state, stress_level, energy_level, stability, alerts)
        """
        alerts = []
        
        # Calculate base scores
        # Stress: deviation from natural patterns
        stress_level = self._calculate_stress(h, d)
        
        # Energy: based on movement complexity
        energy_level = self._calculate_energy(d)
        
        # Stability: how consistent/predictable
        stability = self._calculate_stability(h, d)
        
        # Classify state
        
        # Check for concerning states first
        if d < self.D_DEPRESSED:
            alerts.append(f"[!] Very low complexity (D={d:.2f}) - possible depression indicator")
            return EmotionalState.DEPRESSED, stress_level, energy_level, stability, alerts
        
        if d > self.D_AGITATED:
            alerts.append(f"[!] High chaos (D={d:.2f}) - emotional agitation")
            return EmotionalState.AGITATED, stress_level, energy_level, stability, alerts
        
        # Check for anxiety (chaotic + unfocused)
        if d > self.D_CALM_HIGH and h < self.H_ANXIOUS:
            alerts.append(f"Elevated anxiety indicators")
            return EmotionalState.ANXIOUS, stress_level, energy_level, stability, alerts
        
        # Check for stress (mild chaos)
        if d > self.D_CALM_HIGH:
            return EmotionalState.STRESSED, stress_level, energy_level, stability, alerts
        
        # Check for fatigue (too smooth)
        if d < self.D_CALM_LOW:
            return EmotionalState.FATIGUED, stress_level, energy_level, stability, alerts
        
        # Check for focus (persistent, natural complexity)
        if h >= self.H_FOCUSED and self.D_CALM_LOW <= d <= self.D_CALM_HIGH:
            return EmotionalState.FOCUSED, stress_level, energy_level, stability, alerts
        
        # Default: calm/normal
        return EmotionalState.CALM, stress_level, energy_level, stability, alerts
    
    def _calculate_stress(self, h: float, d: float) -> float:
        """
        Calculate stress level from fractal signature.
        
        Higher D (chaos) and lower H (unpredictability) = more stress
        """
        # D contribution: high D = stress
        d_stress = max(0, (d - 1.5) / 0.5)  # Normalized, 0 at D=1.5
        
        # H contribution: low H = stress (mind is scattered)
        h_stress = max(0, (0.55 - h) / 0.3)  # Normalized, 0 at H=0.55
        
        # Combined
        stress = (d_stress * 0.6 + h_stress * 0.4)
        return min(1.0, max(0.0, stress))
    
    def _calculate_energy(self, d: float) -> float:
        """
        Calculate energy level from complexity.
        
        Natural complexity = healthy energy
        Too low = fatigued, too high = hyperactive
        """
        # Peak energy around D=1.5 (natural)
        # Falls off in both directions
        optimal_d = 1.5
        deviation = abs(d - optimal_d)
        energy = 1.0 - (deviation / 0.5)
        return min(1.0, max(0.0, energy))
    
    def _calculate_stability(self, h: float, d: float) -> float:
        """
        Calculate emotional stability score.
        
        High H (consistent) + natural D = stable
        """
        # H contribution: high H = stable
        h_stability = min(1.0, h / 0.6)
        
        # D contribution: natural range = stable
        if 1.3 <= d <= 1.7:
            d_stability = 1.0
        else:
            d_stability = max(0, 1.0 - abs(d - 1.5) / 0.5)
        
        return (h_stability * 0.5 + d_stability * 0.5)
    
    # =========================================================================
    # Mouse/Trackpad Analysis
    # =========================================================================
    
    def analyze_movement(
        self,
        x: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> MentalStateAnalysis:
        """
        Analyze mouse or trackpad movement for emotional state.
        
        Args:
            x: X coordinates of movement
            y: Y coordinates of movement
            timestamps: Optional timestamps (for velocity analysis)
            
        Returns:
            MentalStateAnalysis with detected state
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if len(x) < 50:
            raise ValueError(f"Movement too short: {len(x)} points. Need at least 50.")
        
        # Calculate velocity/displacement signal
        dx = np.diff(x)
        dy = np.diff(y)
        displacement = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle changes (for jitter detection)
        angles = np.arctan2(dy, dx)
        angle_changes = np.abs(np.diff(angles))
        # Wrap around pi
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        
        # Analyze displacement pattern
        if len(displacement) >= 20:
            h_disp, _ = hurst_exponent(displacement)
            d_disp, _ = higuchi_fractal_dimension(displacement)
        else:
            h_disp, d_disp = 0.5, 1.5
        
        # Analyze jitter (angle changes)
        if len(angle_changes) >= 20:
            d_jitter, _ = higuchi_fractal_dimension(angle_changes)
        else:
            d_jitter = 1.5
        
        # Combined metrics (weight toward displacement)
        h = h_disp
        d = (d_disp * 0.7 + d_jitter * 0.3)
        
        # Get full signature
        signature = self.analyzer.analyze(displacement)
        
        # Classify
        state, stress, energy, stability, alerts = self.classify_state(h, d)
        
        return MentalStateAnalysis(
            signature=signature,
            state=state,
            stress_level=stress,
            energy_level=energy,
            stability_score=stability,
            alerts=alerts
        )
    
    # =========================================================================
    # Touch Analysis
    # =========================================================================
    
    def analyze_touch(
        self,
        pressure: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> MentalStateAnalysis:
        """
        Analyze touch patterns (pressure and movement) for emotional state.
        
        Args:
            pressure: Touch pressure values over time
            x, y: Optional touch coordinates
            
        Returns:
            MentalStateAnalysis
        """
        pressure = np.asarray(pressure, dtype=np.float64)
        
        if len(pressure) < 30:
            raise ValueError(f"Touch data too short: {len(pressure)}. Need at least 30.")
        
        # Analyze pressure pattern
        h_pressure, _ = hurst_exponent(pressure)
        d_pressure, _ = higuchi_fractal_dimension(pressure)
        
        # If coordinates provided, analyze movement too
        if x is not None and y is not None and len(x) >= 30:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            
            dx = np.diff(x)
            dy = np.diff(y)
            displacement = np.sqrt(dx**2 + dy**2)
            
            d_movement, _ = higuchi_fractal_dimension(displacement)
            
            # Combine pressure and movement
            d = (d_pressure * 0.6 + d_movement * 0.4)
        else:
            d = d_pressure
        
        h = h_pressure
        
        signature = self.analyzer.analyze(pressure)
        state, stress, energy, stability, alerts = self.classify_state(h, d)
        
        return MentalStateAnalysis(
            signature=signature,
            state=state,
            stress_level=stress,
            energy_level=energy,
            stability_score=stability,
            alerts=alerts
        )
    
    # =========================================================================
    # Voice Analysis
    # =========================================================================
    
    def analyze_voice_pitch(
        self,
        pitch_values: np.ndarray,
        sample_rate: Optional[float] = None
    ) -> MentalStateAnalysis:
        """
        Analyze voice pitch variation for emotional state.
        
        Morning voice, tremor, and pitch stability reveal emotional state.
        
        Args:
            pitch_values: F0 (fundamental frequency) values over time
            sample_rate: Optional sample rate in Hz
            
        Returns:
            MentalStateAnalysis
        """
        pitch = np.asarray(pitch_values, dtype=np.float64)
        
        # Remove zeros/NaN (unvoiced segments)
        pitch = pitch[pitch > 0]
        pitch = pitch[~np.isnan(pitch)]
        
        if len(pitch) < 30:
            raise ValueError(f"Not enough voiced samples: {len(pitch)}. Need at least 30.")
        
        # Analyze pitch variation
        h, _ = hurst_exponent(pitch)
        d, _ = higuchi_fractal_dimension(pitch)
        
        signature = self.analyzer.analyze(pitch)
        state, stress, energy, stability, alerts = self.classify_state(h, d)
        
        # Additional voice-specific analysis
        pitch_std = np.std(pitch)
        pitch_mean = np.mean(pitch)
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
        
        # High pitch variation may indicate stress
        if pitch_cv > 0.15:
            stress = min(1.0, stress + 0.2)
            if "stress" not in str(alerts).lower():
                alerts.append(f"High pitch variability (CV={pitch_cv:.2f})")
        
        # Tremor detection (rapid oscillations)
        pitch_diff = np.diff(pitch)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(pitch_diff))) > 0)
        tremor_rate = zero_crossings / len(pitch_diff)
        
        if tremor_rate > 0.4:
            stress = min(1.0, stress + 0.15)
            alerts.append(f"Possible voice tremor detected")
        
        return MentalStateAnalysis(
            signature=signature,
            state=state,
            stress_level=stress,
            energy_level=energy,
            stability_score=stability,
            alerts=alerts
        )
    
    # =========================================================================
    # Tracking Over Time
    # =========================================================================
    
    def track_state_changes(
        self,
        analyses: List[MentalStateAnalysis]
    ) -> Dict:
        """
        Track mental state changes over a series of analyses.
        
        Returns:
            Dictionary with trend information
        """
        if len(analyses) < 2:
            return {"trend": "insufficient_data", "changes": []}
        
        # Track stress trend
        stress_values = [a.stress_level for a in analyses]
        stress_trend = stress_values[-1] - stress_values[0]
        
        # Track complexity trend
        d_values = [a.signature.higuchi for a in analyses]
        d_trend = d_values[-1] - d_values[0]
        
        # Detect state transitions
        transitions = []
        for i in range(1, len(analyses)):
            if analyses[i].state != analyses[i-1].state:
                transitions.append({
                    "index": i,
                    "from": analyses[i-1].state.value,
                    "to": analyses[i].state.value
                })
        
        # Determine overall trend
        if stress_trend > 0.2:
            trend = "increasing_stress"
        elif stress_trend < -0.2:
            trend = "decreasing_stress"
        elif abs(d_trend) > 0.2:
            trend = "changing_energy"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "stress_change": round(stress_trend, 3),
            "complexity_change": round(d_trend, 3),
            "transitions": transitions,
            "avg_stress": round(np.mean(stress_values), 3),
            "avg_stability": round(np.mean([a.stability_score for a in analyses]), 3)
        }


# =============================================================================
# Demo / Test
# =============================================================================

def generate_smooth_movement(n: int, smoothness: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic mouse movement by filtering random walk.
    
    Args:
        n: Number of points
        smoothness: 0-1, higher = smoother movement
    """
    # Start with random walk (cumsum of small steps)
    raw_x = np.cumsum(np.random.randn(n))
    raw_y = np.cumsum(np.random.randn(n))
    
    # Apply exponential moving average for smoothness
    alpha = 1 - smoothness
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = raw_x[0], raw_y[0]
    
    for i in range(1, n):
        x[i] = alpha * raw_x[i] + (1 - alpha) * x[i-1]
        y[i] = alpha * raw_y[i] + (1 - alpha) * y[i-1]
    
    return x * 10, y * 10  # Scale up


if __name__ == "__main__":
    print("=" * 60)
    print("BIO ADAPTER - Mental Health Biometrics Demo")
    print("=" * 60)
    
    np.random.seed(42)
    adapter = BioAdapter()
    n = 300
    
    # 1. Calm/Natural Mouse Movement (smooth, consistent)
    print("\n[1] Calm Mouse Movement (smooth cursor)")
    x_calm, y_calm = generate_smooth_movement(n, smoothness=0.9)
    
    analysis = adapter.analyze_movement(x_calm, y_calm)
    print(f"  State: {analysis.state.value}")
    print(f"  Stress: {analysis.stress_level:.0%}, Energy: {analysis.energy_level:.0%}")
    print(f"  H={analysis.signature.hurst:.3f}, D={analysis.signature.higuchi:.3f}")
    
    # 2. Anxious/Jittery Movement (less smooth, more random)
    print("\n[2] Anxious/Jittery Movement")
    x_anx, y_anx = generate_smooth_movement(n, smoothness=0.3)
    # Add extra jitter
    x_anx += np.random.randn(n) * 5
    y_anx += np.random.randn(n) * 5
    
    analysis = adapter.analyze_movement(x_anx, y_anx)
    print(f"  State: {analysis.state.value}")
    print(f"  Stress: {analysis.stress_level:.0%}")
    print(f"  H={analysis.signature.hurst:.3f}, D={analysis.signature.higuchi:.3f}")
    
    # 3. Fatigued/Sluggish Movement (very smooth, slow)
    print("\n[3] Fatigued/Sluggish Movement (very slow, smooth)")
    x_fat, y_fat = generate_smooth_movement(n, smoothness=0.98)
    
    analysis = adapter.analyze_movement(x_fat, y_fat)
    print(f"  State: {analysis.state.value}")
    print(f"  Energy: {analysis.energy_level:.0%}")
    print(f"  D={analysis.signature.higuchi:.3f} (low = fatigued)")
    
    # 4. Focused Movement (consistent direction, natural complexity)
    print("\n[4] Focused Movement (purposeful)")
    t = np.linspace(0, 10, n)
    x_focus = 50 * t + np.cumsum(np.random.randn(n) * 0.5)  # Strong trend
    y_focus = 30 * t + np.cumsum(np.random.randn(n) * 0.4)
    
    analysis = adapter.analyze_movement(x_focus, y_focus)
    print(f"  State: {analysis.state.value}")
    print(f"  H={analysis.signature.hurst:.3f} (high = focused)")
    print(f"  Stability: {analysis.stability_score:.0%}")
    
    # 5. Touch Pressure Analysis
    print("\n[5] Touch Pressure (Calm vs Stressed)")
    
    # Calm: consistent pressure with natural variation
    pressure_calm = 50 + np.cumsum(np.random.randn(150) * 0.5)
    analysis_calm = adapter.analyze_touch(pressure_calm)
    print(f"  Calm: {analysis_calm.state.value}, Stress={analysis_calm.stress_level:.0%}")
    
    # Stressed: erratic pressure
    pressure_stress = 50 + np.cumsum(np.random.randn(150) * 2) + np.random.randn(150) * 5
    analysis_stress = adapter.analyze_touch(pressure_stress)
    print(f"  Stressed: {analysis_stress.state.value}, Stress={analysis_stress.stress_level:.0%}")
    
    # 6. Voice Pitch Analysis  
    print("\n[6] Voice Pitch (Stable vs Tremor)")
    
    # Stable pitch: normal variation
    pitch_stable = 120 + np.cumsum(np.random.randn(150) * 0.3)
    analysis = adapter.analyze_voice_pitch(pitch_stable)
    print(f"  Stable: {analysis.state.value}")
    
    # Tremor: rapid oscillations
    t = np.linspace(0, 3, 150)
    pitch_tremor = 120 + np.cumsum(np.random.randn(150) * 0.3) + 5 * np.sin(t * 40)
    analysis = adapter.analyze_voice_pitch(pitch_tremor)
    print(f"  Tremor: {analysis.state.value}, Alerts: {analysis.alerts}")
    
    # 7. State Tracking
    print("\n[7] State Tracking Throughout Day")
    analyses = []
    smoothness_levels = [0.9, 0.85, 0.7, 0.5, 0.4]  # Decreasing calm
    for smooth in smoothness_levels:
        x, y = generate_smooth_movement(n, smoothness=smooth)
        analyses.append(adapter.analyze_movement(x, y))
    
    tracking = adapter.track_state_changes(analyses)
    print(f"  Trend: {tracking['trend']}")
    print(f"  Stress Change: +{tracking['stress_change']:.0%}")
    print(f"  State transitions: {len(tracking['transitions'])}")
    for t in tracking['transitions']:
        print(f"    {t['from']} -> {t['to']}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  CALM: Natural movement, D~1.4-1.6")
    print("  ANXIOUS: Jittery, D>1.65, H<0.4")
    print("  FATIGUED: Overly smooth, D<1.35")
    print("  FOCUSED: High persistence, H>0.65")
    print("=" * 60)

