# 🎓 Sorcerry Project Interview Guide

This guide is designed to help you explain this project confidently in a job interview, even if you are a beginner. **Don't memorize this—understand the "Why".**

---

## 1. The 30-Second Pitch (The "Hook")
**Interviewer:** "Tell me about this project."

**You:** 
> "Sorcerry is a **headless, biometric tracking system** for mental health. 
> Unlike typical apps that ask you 'how do you feel?', it passively analyzes your **mouse movements** and **voice pitch** in the background.
> Instead of using heavy Machine Learning models, I used **Fractal Geometry (Chaos Theory)** to detect stress patterns. This allows it to run locally on any device with near-zero latency and instant personalization."

---

## 2. The Core Concept: "Why Fractals?"
**Interviewer:** "Why didn't you just use AI/ML?"

**You:**
> "Three reasons:
> 1.  **Data Efficiency**: Neural Networks need thousands of examples to learn. My system learns your baseline in **30 seconds**.
> 2.  **Privacy**: AI models often need cloud processing. My math runs 100% locally on the CPU.
> 3.  **Sensitivity**: Stress shows up as 'micro-tremors'. Fractal analysis (Hurst Exponent) is calculated specifically to measure rough/jittery time-series data, which is perfect for biological signals."

---

## 3. System Architecture (How it works)
**Interviewer:** "How is it built?"

**You:**
> "It has three main parts decoupled for performance:
> 1.  **The Daemon (Python)**: A background service that uses `ctypes` to tap into Windows API for low-level mouse tracking. It buffers data and runs the math every 15 seconds.
> 2.  **The API (Python `http.server`)**: A lightweight REST API that serves the data. I avoided heavy frameworks like Django to keep it minimal.
> 3.  **The Dashboard (Vanilla JS)**: A frontend that visualizes the data. I used pure JavaScript and Canvas API for the charts to ensure it's fast and dependency-free."

---

## 4. Key Technical Challenges & Solutions
**Interviewer:** "What was the hardest part?"

**(Pick one of these to talk about):**

### Challenge A: "Signal vs. Noise in Mouse Data"
*   **Problem**: Mice are sensitive. A tiny movement might look like stress.
*   **Solution**: I implemented a **Velocity Threshold**. I only analyze movements where speed > 5 pixels/sec. This filters out accidental jitters and focuses on *intentional* motion.

### Challenge B: "Persisting Data without a Database"
*   **Problem**: I didn't want to force users to install SQL/Mongo.
*   **Solution**: I built a **Custom JSON Storage Engine**. Data is sharded by date (`history_2024-03-20.json`). I also implemented a **Fractal Memory** algorithm that compresses old data by keeping only "significant" events (high stress moments) and averaging the rest.

### Challenge C: "Real-time Voice Analysis"
*   **Problem**: Audio processing is usually slow.
*   **Solution**: I used `numpy` for fast array operations and an **Autocorrelation** algorithm (not FFT) to estimate pitch, which works efficiently even on short, noisy clips.

---

## 5. Code Spotlight (Be ready to explain this!)
If they ask to see code, show them `BioAdapter.analyze_movement` (in `adapters/bio_adapter.py`).

**Explanation:**
> "This function takes a list of (x, y) coordinates.
> First, it calculates the **velocity profile** (how fast you moved between points).
> Then, it calculates the **Higuchi Dimension (D)**. Ideally, a straight line is D=1. A shaky, stressed hand approaches D=2.
> If D > 1.6, the system flags it as 'High Cognitive Load'."

---

## 6. Glossary (Cheat Sheet)

*   **Hurst Exponent (H)**: A number from 0 to 1. 0.5 is random. >0.5 is smooth/trend. <0.5 is jittery/mean-reverting. **We want High H**.
*   **Higuchi Dimension (D)**: Roughness. 1.0 is smooth. 2.0 is chaos. **We want Low D**.
*   **Daemon**: A computer program that runs as a background process rather than being under the direct control of an interactive user.
*   **Headless**: Software capable of working on a device without a graphical user interface.
*   **Autocorrelation**: Comparing a signal with a delayed copy of itself to find repeating patterns (pitch).

---

**Good luck! You built something real and complex. Be proud.**
