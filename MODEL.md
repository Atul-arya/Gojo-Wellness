# 🧙‍♂️ The Fractal Analysis Model (Sorcerry)

This system uses **Chaos Theory** and **Fractal Geometry** to quantify your mental state by analyzing the roughness and memory of time-series data (your movements and voice).

It does NOT use AI (Neural Networks). It uses pure mathematics.

## 1. The Core Metrics

### H (Hurst Exponent): Long-Term Memory
- **What it is**: Measure of how "smooth" or "persistent" a time series is.
- **Range**: 0.0 to 1.0 (0.5 = Random Walk/Brownian Motion).
- **Interpretation**: 
  - **H > 0.5 (Persistent)**: Clean, deliberate movements. **Focus/Flow**.
  - **H < 0.5 (Anti-Persistent)**: Jittery, rapidly changing direction. **Anxiety/Stress**.
  - **H = 0.5**: Random. **Fatigue**.

### D (Higuchi Fractal Dimension): Roughness
- **What it is**: Measure of how much "space" a curve fills.
- **Range**: 1.0 (Line) to 2.0 (Plane).
- **Interpretation**:
  - **D ~ 1.0**: Very smooth, simple movements. **Calm**.
  - **D > 1.5**: Highly complex, jagged movements. **High Cognitive Load / Stress**.

## 2. The Bio-Mapping

We extract time-series data from two sources:

### A. Mouse Movements (Continuous)
- We sample (x, y) coordinates at 20Hz.
- We calculate the velocity vector $V(t)$.
- We compute $H$ and $D$ on the velocity series.
- **Why?** Stressed muscles have micro-tremors (higher $D$) and erratic corrections (lower $H$).

### B. Voice Pitch (Snapshot)
- We extract the Fundamental Frequency ($F_0$) contour over time.
- A "monotone" voice has low $D$ (flat).
- A "stressed" voice has high jitter ($D$) and erratic changes ($H$).
- **Depression biomarker**: Very low $H$, low $D$ (flat affect).
- **Anxiety biomarker**: High $D$ (tremor), low $H$ (instability).

## 3. The Personal Baseline (The "Secret Sauce")

Most systems fail because they use population averages (e.g., "Heart rate > 100 is high").
Your baseline is unique.

1.  **Calibration**: We record your "neutral" state for 5 minutes.
2.  **Learning**: We compute your personal mean ($\mu$) and standard deviation ($\sigma$) for $H$ and $D$.
3.  **Z-Score Tracking**: We track how far you deviate from YOUR normal.
    $$Z = \frac{\text{Current} - \mu}{\sigma}$$
    - If $Z > +2$: Significant deviation (Alert).

## 4. The Insight Engine

We combine these metrics into human-readable states:

| Condition | H (Memory) | D (Roughness) | State |
| :--- | :--- | :--- | :--- |
| High Focus | High (>0.6) | Low (<1.2) | **FLOW** 🌊 |
| High Stress | Low (<0.4) | High (>1.6) | **ANXIOUS** ⚡ |
| Burnout | Low (<0.4) | Low (<1.1) | **FATIGUED** 🛑 |
| Normal | ~0.5 | ~1.3 | **BALANCED** ⚖️ |

This allows the system to predict burnout *before* you feel it, by detecting subtle degradation in your motor control and vocal patterns.
