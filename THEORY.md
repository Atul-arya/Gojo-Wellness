# 🧪 The Mathematical Theory of Sorcerry (Project Gojo)

This document provides a deep dive into the mathematical models used to analyze human biometric data and detect mental states.

---

## 1. Hurst Exponent ($H$)
The Hurst exponent measures the **long-term memory** or "persistence" of a time series.

### Method: Detrended Fluctuation Analysis (DFA)
Instead of the classical Rescaled Range (R/S) analysis, we use DFA because it is more robust to non-stationary data (data whose mean or variance changes over time, like human movement).

**Step 1: Integration**
Convert the original signal $x(i)$ into an integrated profile $y(k)$:
$$y(k) = \sum_{i=1}^k (x(i) - \bar{x})$$

**Step 2: Partitioning & Local Detrending**
Divide the profile into segments of length $n$. For each segment, calculate the local RMS fluctuation $F(n)$ by fitting a linear trend $y_n(k)$ and subtracting it:
$$F(n) = \sqrt{\frac{1}{N} \sum_{k=1}^N [y(k) - y_n(k)]^2}$$

**Step 3: Scaling Law**
The fluctuation $F(n)$ scales according to a power law:
$$F(n) \propto n^H$$

By plotting $\log(F(n))$ against $\log(n)$, the slope of the line gives us the **Hurst Exponent ($H$)**.

---

## 2. Higuchi Fractal Dimension ($D$)
The Higuchi Dimension measures the **complexity** or "roughness" of the signal in the time domain.

### Algorithm
**Step 1: Create Sub-series**
From a series $X = \{X(1), X(2), \dots, X(N)\}$, we construct $k$ new time series:
$$X_m^k = \{X(m), X(m+k), X(m+2k), \dots, X(m + \lfloor \frac{N-m}{k} \rfloor k)\}$$
where $m = 1, 2, \dots, k$.

**Step 2: Calculate Length**
The length $L_m(k)$ of each sub-series is calculated as:
$$L_m(k) = \frac{\sum_{i=1}^{\lfloor \frac{N-m}{k} \rfloor} |X(m+ik) - X(m+(i-1)k)| \cdot (N-1)}{\lfloor \frac{N-m}{k} \rfloor \cdot k^2}$$

**Step 3: Total Length**
The average length $L(k)$ for a scale $k$ is:
$$L(k) = \frac{1}{k} \sum_{m=1}^k L_m(k)$$

**Step 4: Dimension Estimation**
$$L(k) \propto k^{-D}$$
The slope of $\log(L(k))$ vs $\log(k)$ gives the **Fractal Dimension ($D$)**.

---

## 3. Biometric Mapping (The Bio-Adapter)
We translate these raw values ($H, D$) into human emotional states using weighted heuristics.

### Base Stress Formula ($S_{base}$)
Stress is calculated by measuring how much current $D$ reflects chaos and $H$ reflects unpredictability:
$$S_{D} = \max(0, \frac{D - 1.5}{0.5})$$
$$S_{H} = \max(0, \frac{0.55 - H}{0.3})$$
$$S_{base} = (0.6 \cdot S_D) + (0.4 \cdot S_H)$$

### Energy Score ($E$)
Energy is maxed at "Natural Complexity" ($D \approx 1.5$):
$$E = 1.0 - \frac{|D - 1.5|}{0.5}$$

---

## 4. Personalization (The Adaptive Filter)
This is where the system "learns" you. We use the **Z-Score** to measure how weird your current state is compared to your own normal.

### Z-Score for Displacement
During calibration, we calculate your mean ($\mu$) and standard deviation ($\sigma$).
$$Z_H = \frac{H - \mu_H}{\sigma_H}$$
$$Z_D = \frac{D - \mu_D}{\sigma_D}$$

### Relative Stress ($S_{rel}$)
We compare your current stress $S_{base}$ to your learned $S_{baseline}$:
$$S_{rel} = \text{clamp}(2 \cdot (S_{base} - S_{baseline}), -1, 1)$$

---

## 5. Summary Table

| Metric | Ideal (Calm) | Stressed | Meaning |
| :--- | :--- | :--- | :--- |
| **Hurst (H)** | 0.5 - 0.7 | < 0.45 | Degree of persistence/control. |
| **Higuchi (D)** | 1.4 - 1.6 | > 1.7 | Degree of roughness/chaos. |
| **Z-Score** | 0.0 | > 2.0 | Statistical deviation from YOUR normal. |
