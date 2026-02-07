"""
Personal Tracker - Adaptive Wellness Monitoring System

This module provides personalized mental health tracking that LEARNS
your individual baseline. An athlete's "calm" is different from a
stressed person's "calm" - this system adapts to YOU.

CORE CONCEPT:
1. CALIBRATION PHASE - Collect your normal patterns for ~5 minutes
2. BASELINE LEARNING - Calculate YOUR personal H/D baseline
3. RELATIVE TRACKING - Compare current state to YOUR baseline
4. LOCAL STORAGE - All data stays on your device

NO TWO PEOPLE ARE THE SAME:
- Athlete: Baseline H~0.7 (focused), D~1.3 (controlled)
- Anxious person: Baseline H~0.5, D~1.7 (their normal)
- We compare YOU to YOUR baseline, not to population averages
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fractal_core import hurst_exponent, higuchi_fractal_dimension
from adapters.bio_adapter import BioAdapter, EmotionalState, MentalStateAnalysis
from fractal_memory import TimeFractalMemory
import audio_utils


@dataclass
class PersonalBaseline:
    """Your personal baseline metrics (learned from YOUR data)."""
    h_mean: float           # Your average Hurst
    h_std: float            # Your Hurst variation
    d_mean: float           # Your average Higuchi D
    d_std: float            # Your D variation
    stress_baseline: float  # Your normal stress level
    samples_collected: int  # How many samples in calibration
    calibrated_at: str      # When calibration was done
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonalBaseline':
        return cls(**data)


@dataclass
class StateSnapshot:
    """A single moment's state measurement."""
    timestamp: str
    h: float
    d: float
    stress_relative: float  # Relative to YOUR baseline (-1 to +1)
    state: str
    deviation: str         # "normal", "elevated", "low"
    tag: Optional[str] = None  # User-defined mood tag
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PersonalTracker:
    """
    Personalized wellness tracker that learns YOUR baseline.
    
    Example:
        >>> tracker = PersonalTracker("my_profile")
        >>> 
        >>> # Calibration (first 5 minutes of use)
        >>> for movement in collect_mouse_movements():
        ...     tracker.add_calibration_sample(x, y)
        >>> tracker.finalize_calibration()
        >>> 
        >>> # Now track relative to YOUR baseline
        >>> state = tracker.analyze_current(x, y)
        >>> print(f"You are {state.deviation} compared to your normal")
    """
    
    def __init__(
        self,
        profile_name: str = "default",
        data_dir: Optional[str] = None
    ):
        """
        Initialize tracker with a profile.
        
        Args:
            profile_name: Name for this person's profile
            data_dir: Directory for storing data (default: ./gojo_data)
        """
        self.profile_name = profile_name
        
        # Set up data directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "gojo_data")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.profile_dir = self.data_dir / profile_name
        self.profile_dir.mkdir(exist_ok=True)
        
        # Initialize bio adapter
        self.bio = BioAdapter()
        
        # Calibration state
        self.calibration_samples: List[Dict] = []
        self.is_calibrated = False
        self.baseline: Optional[PersonalBaseline] = None
        
        # Session history
        self.session_history: List[StateSnapshot] = []
        
        # Load existing profile if exists
        self._load_profile()
        
        # Load today's history if exists (prevent overwrite on restart)
        self._load_daily_history()
    
    # =========================================================================
    # Profile Persistence
    # =========================================================================
    
    def _get_profile_path(self) -> Path:
        return self.profile_dir / "baseline.json"
    
    def _get_history_path(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.profile_dir / f"history_{date_str}.json"
    
    def _load_profile(self):
        """Load existing profile from disk."""
        profile_path = self._get_profile_path()
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                data = json.load(f)
                self.baseline = PersonalBaseline.from_dict(data)
                self.is_calibrated = True
                
    def _load_daily_history(self):
        """Load today's history to resume session."""
        path = self._get_history_path()
        if path.exists():
            try:
                if path.stat().st_size == 0:
                    return
                with open(path, 'r') as f:
                    content = f.read()
                    if not content:
                        return
                    data = json.loads(content)
                    # Handle both old and new format if needed, but assuming list of dicts
                    if isinstance(data, list):
                        self.session_history = []
                        for item in data:
                            try:
                                # Filter out keys that aren't in StateSnapshot fields if needed
                                # or just let **item expand. 
                                # If item has extra keys, StateSnapshot init might fail if not strict?
                                # dataclass init does not accept extra keys.
                                valid_keys = StateSnapshot.__annotations__.keys()
                                filtered = {k: v for k, v in item.items() if k in valid_keys}
                                self.session_history.append(StateSnapshot(**filtered))
                            except Exception:
                                pass
            except Exception as e:
                print(f"Error loading daily history: {e}")

    def _save_profile(self):
        """Save profile to disk."""
        if self.baseline:
            with open(self._get_profile_path(), 'w') as f:
                json.dump(self.baseline.to_dict(), f, indent=2)
    
    def _save_history(self):
        """Save today's session history."""
        history_path = self._get_history_path()
        
        # Load existing history for today
        existing = []
        if history_path.exists():
            with open(history_path, 'r') as f:
                existing = json.load(f)
        
        # Append new
        existing.extend([s.to_dict() for s in self.session_history])
        
        with open(history_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        self.session_history = []
    
    # =========================================================================
    # Calibration
    # =========================================================================
    
    def add_calibration_sample(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Add a movement sample during calibration.
        
        Collect samples for 3-5 minutes during normal activity.
        
        Returns:
            Current calibration status
        """
        if len(x) < 50:
            return {"status": "need_more_data", "samples": len(self.calibration_samples)}
        
        try:
            analysis = self.bio.analyze_movement(x, y)
            self.calibration_samples.append({
                "h": analysis.signature.hurst,
                "d": analysis.signature.higuchi,
                "stress": analysis.stress_level
            })
        except ValueError:
            pass
        
        return {
            "status": "collecting",
            "samples": len(self.calibration_samples),
            "ready": len(self.calibration_samples) >= 10
        }
    
    def finalize_calibration(self) -> PersonalBaseline:
        """
        Complete calibration and compute YOUR personal baseline.
        
        Should have at least 10 samples.
        """
        if len(self.calibration_samples) < 5:
            raise ValueError(f"Need at least 5 samples, have {len(self.calibration_samples)}")
        
        h_values = [s["h"] for s in self.calibration_samples]
        d_values = [s["d"] for s in self.calibration_samples]
        stress_values = [s["stress"] for s in self.calibration_samples]
        
        self.baseline = PersonalBaseline(
            h_mean=float(np.mean(h_values)),
            h_std=float(np.std(h_values)),
            d_mean=float(np.mean(d_values)),
            d_std=float(np.std(d_values)),
            stress_baseline=float(np.mean(stress_values)),
            samples_collected=len(self.calibration_samples),
            calibrated_at=datetime.now().isoformat()
        )
        
        self.is_calibrated = True
        self._save_profile()
        self.calibration_samples = []
        
        return self.baseline
    
    def reset_calibration(self):
        """Reset and start fresh calibration."""
        self.calibration_samples = []
        self.baseline = None
        self.is_calibrated = False
        
        profile_path = self._get_profile_path()
        if profile_path.exists():
            os.remove(profile_path)
    
    # =========================================================================
    # Real-time Analysis (Relative to YOUR Baseline)
    # =========================================================================
    
    def analyze_current(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> StateSnapshot:
        """
        Analyze current state RELATIVE TO YOUR BASELINE.
        
        This is the key feature: we compare YOU to YOU, not to averages.
        """
        if not self.is_calibrated:
            raise ValueError("Not calibrated. Run calibration first.")
        
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Get raw analysis
        analysis = self.bio.analyze_movement(x, y)
        h = analysis.signature.hurst
        d = analysis.signature.higuchi
        
        # Calculate deviation from YOUR baseline
        h_z = (h - self.baseline.h_mean) / max(self.baseline.h_std, 0.01)
        d_z = (d - self.baseline.d_mean) / max(self.baseline.d_std, 0.01)
        
        # Stress relative to YOUR normal (-1 to +1 scale)
        stress_relative = (analysis.stress_level - self.baseline.stress_baseline)
        stress_relative = max(-1, min(1, stress_relative * 2))
        
        # Determine deviation category
        combined_z = abs(h_z) + abs(d_z)
        if combined_z < 1.5:
            deviation = "normal"
        elif d_z > 1 or stress_relative > 0.3:
            deviation = "elevated"
        else:
            deviation = "low"
        
        snapshot = StateSnapshot(
            timestamp=datetime.now().isoformat(),
            h=round(h, 4),
            d=round(d, 4),
            stress_relative=round(stress_relative, 3),
            state=analysis.state.value,
            deviation=deviation
        )
        
        self.session_history.append(snapshot)
        
        # Auto-save every 50 snapshots
        if len(self.session_history) >= 50:
            self._save_history()
        
        return snapshot
    
        return snapshot
    
    def analyze_voice(self, audio_bytes: bytes) -> Dict:
        """
        Analyze voice recording for burnout/depression markers.
        
        Args:
            audio_bytes: Raw WAV audio data
            
        Returns:
            Analysis dictionary
        """
        # Load audio using our utils
        audio, sr = audio_utils.load_audio_from_bytes(audio_bytes)
        
        if len(audio) == 0:
            return {"error": "Could not decode audio"}
            
        # Extract pitch
        pitch = audio_utils.extract_pitch(audio, sr)
        
        if len(pitch) < 30:
            return {"error": "Voice not detected (too quiet or short)"}
            
        # Analyze using BioAdapter
        try:
            analysis = self.bio.analyze_voice_pitch(pitch, sr)
            
            # Create a snapshot-like dict
            result = {
                "timestamp": datetime.now().isoformat(),
                "state": analysis.state.value,
                "stress": analysis.stress_level,
                "energy": analysis.energy_level,
                "h": analysis.signature.hurst,
                "d": analysis.signature.higuchi,
                "alerts": analysis.alerts,
                "pitch_avg": float(np.mean(pitch)) if len(pitch) > 0 else 0
            }
            
            # We can save this to a separate voice log later if needed
            return result
            
        except Exception as e:
            return {"error": str(e)}

    def get_session_summary(self) -> Dict:
        """Get summary of current session."""
        if not self.session_history:
            return {"status": "no_data"}
        
        stress_vals = [s.stress_relative for s in self.session_history]
        states = [s.state for s in self.session_history]
        deviations = [s.deviation for s in self.session_history]
        
        return {
            "snapshots": len(self.session_history),
            "avg_stress_relative": round(np.mean(stress_vals), 3),
            "max_stress_relative": round(max(stress_vals), 3),
            "most_common_state": max(set(states), key=states.count),
            "elevated_pct": round(deviations.count("elevated") / len(deviations) * 100, 1),
            "baseline_h": round(self.baseline.h_mean, 3) if self.baseline else None,
            "baseline_d": round(self.baseline.d_mean, 3) if self.baseline else None
        }
    
    def end_session(self):
        """End session and save all data."""
        self._save_history()
    
    # =========================================================================
    # Historical Analysis
    # =========================================================================
    
    def get_daily_history(self, date_str: Optional[str] = None) -> List[Dict]:
        """Load history for a specific day."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        history_path = self.profile_dir / f"history_{date_str}.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return []
    
    def get_weekly_summary(self) -> Dict:
        """Get summary of the past 7 days."""
        from datetime import timedelta
        
        all_data = []
        today = datetime.now()
        
        for i in range(7):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            daily = self.get_daily_history(date_str)
            for entry in daily:
                entry["date"] = date_str
                all_data.append(entry)
        
        if not all_data:
            return {"status": "no_data", "days": 0}
        
        stress_vals = [d.get("stress_relative", 0) for d in all_data]
        elevated_count = sum(1 for d in all_data if d.get("deviation") == "elevated")
        
        return {
            "days": 7,
            "total_snapshots": len(all_data),
            "avg_stress": round(np.mean(stress_vals), 3),
            "max_stress": round(max(stress_vals), 3),
            "elevated_pct": round(elevated_count / len(all_data) * 100, 1),
            "trend": "improving" if stress_vals[-10:] < stress_vals[:10] else "stable"
        }
    
    # =========================================================================
    # Archival / Fractal Memory Storage
    # =========================================================================

    def archive_old_history(self, days_to_keep: int = 7) -> Dict:
        """
        Compress old history files using TimeFractalMemory.
        
        Args:
            days_to_keep: Number of recent days to keep raw
            
        Returns:
            Summary of archival
        """
        memory = TimeFractalMemory()
        archived_count = 0
        points_removed = 0
        
        # Ensure archive dir exists
        archive_dir = self.profile_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        cutoff_date = datetime.now().strftime("%Y-%m-%d") # Compare strings for simplicity
        # Ideally parsing, but ISO format sorts lexically so it's fine
        
        # List all history files
        for file_path in self.profile_dir.glob("history_*.json"):
            # Check if it's already in archive (it shouldn't be as we move it)
            filename = file_path.name
            file_date = filename.replace("history_", "").replace(".json", "")
            
            # Simple string comparison for "older than x days" is tricky
            # Let's parse
            try:
                f_date = datetime.strptime(file_date, "%Y-%m-%d")
                days_diff = (datetime.now() - f_date).days
                
                if days_diff >= days_to_keep:
                    # Compress!
                    with open(file_path, 'r') as f:
                        raw_data = json.load(f)
                    
                    if not raw_data:
                        continue
                        
                    compressed = memory.compress_history(raw_data)
                    
                    # Save to archive
                    archive_path = archive_dir / filename
                    with open(archive_path, 'w') as f:
                        json.dump(compressed, f, indent=2)
                        
                    # Remove original
                    os.remove(file_path)
                    
                    archived_count += 1
                    points_removed += (len(raw_data) - len(compressed))
                    
            except Exception as e:
                print(f"Error archiving {filename}: {e}")
                continue
                
        return {
            "status": "success",
            "files_archived": archived_count,
            "points_removed": points_removed
        }
    
    # =========================================================================
    # Phase 8: Fractal Insight Engine
    # =========================================================================

    def tag_current_state(self, tag: str) -> bool:
        """Tag the current moment with a user mood."""
        if not self.session_history:
            return False
            
        # Tag the last snapshot
        self.session_history[-1].tag = tag
        
        # Force save to ensure tag is persisted
        self._save_history()
        return True
        
    def get_insight(self) -> Dict:
        """
        Generate insights based on fractal pattern matching.
        """
        if not self.session_history:
            return {"status": "no_data"}
            
        current = self.session_history[-1]
        
        # Load recent history for context (last 7 days)
        # In a real system, we might cache this or use the archive
        history_context = []
        try:
            # Just grab the last ~5 days
            for file_path in sorted(self.profile_dir.glob("history_*.json"))[-5:]:
                with open(file_path, 'r') as f:
                    history_context.extend(json.load(f))
        except Exception:
            pass
            
        # Search for matches
        memory = TimeFractalMemory()
        matches = memory.search_pattern(history_context, current.h, current.d, tolerance=0.1)
        
        # Analyze matches for predictive alerts
        alert = None
        predicted_mood = "Unknown"
        
        if matches:
            # Look for tags in matches
            tags = [m.get('tag') for m in matches if m.get('tag')]
            if tags:
                from collections import Counter
                predicted_mood = Counter(tags).most_common(1)[0][0]
                
            # Check for bad outcomes in matching patterns
            bad_outcomes = [m for m in matches if m.get('stress_relative', 0) > 0.5]
            if len(bad_outcomes) > len(matches) * 0.3:
                alert = "⚠️ Caution: Similar patterns often led to high stress."
        
        return {
            "current_pattern": {"h": current.h, "d": current.d},
            "similar_moments": len(matches),
            "predicted_mood": predicted_mood,
            "alert": alert,
            "match_sample": matches[:3] if matches else []
        }


# =============================================================================
# Quick API for Dashboard
# =============================================================================

def create_tracker(profile_name: str = "default") -> PersonalTracker:
    """Create a new personal tracker."""
    return PersonalTracker(profile_name)


def quick_calibration_demo():
    """Demo of calibration process."""
    print("=" * 60)
    print("PERSONAL TRACKER - Calibration Demo")
    print("=" * 60)
    
    np.random.seed(42)
    tracker = PersonalTracker("demo_user")
    
    # Simulate calibration (collecting 15 samples)
    print("\n[1] Simulating Calibration...")
    n = 300
    
    for i in range(15):
        # Simulate mouse movement with this person's characteristic pattern
        x = np.cumsum(np.random.randn(n) * 0.8)  # Their normal smoothness
        y = np.cumsum(np.random.randn(n) * 0.8)
        result = tracker.add_calibration_sample(x, y)
        
    print(f"  Collected {result['samples']} samples")
    
    # Finalize
    baseline = tracker.finalize_calibration()
    print(f"\n[2] YOUR Baseline Learned:")
    print(f"  Your H: {baseline.h_mean:.3f} +/- {baseline.h_std:.3f}")
    print(f"  Your D: {baseline.d_mean:.3f} +/- {baseline.d_std:.3f}")
    print(f"  Your stress baseline: {baseline.stress_baseline:.0%}")
    
    # Now analyze relative to baseline
    print("\n[3] Testing Relative Analysis...")
    
    # Normal state (similar to calibration)
    x_normal = np.cumsum(np.random.randn(n) * 0.8)
    y_normal = np.cumsum(np.random.randn(n) * 0.8)
    state = tracker.analyze_current(x_normal, y_normal)
    print(f"  Normal movement: {state.deviation} (stress relative: {state.stress_relative:+.0%})")
    
    # Stressed state (more jittery than YOUR normal)
    x_stressed = np.cumsum(np.random.randn(n) * 1.5) + np.random.randn(n) * 3
    y_stressed = np.cumsum(np.random.randn(n) * 1.5) + np.random.randn(n) * 3
    state = tracker.analyze_current(x_stressed, y_stressed)
    print(f"  Stressed movement: {state.deviation} (stress relative: {state.stress_relative:+.0%})")
    
    # Calm state (smoother than YOUR normal)
    x_calm = np.cumsum(np.random.randn(n) * 0.3)
    y_calm = np.cumsum(np.random.randn(n) * 0.3)
    state = tracker.analyze_current(x_calm, y_calm)
    print(f"  Calm movement: {state.deviation} (stress relative: {state.stress_relative:+.0%})")
    
    # Session summary
    print("\n[4] Session Summary:")
    summary = tracker.get_session_summary()
    print(f"  Snapshots: {summary['snapshots']}")
    print(f"  Avg stress (relative): {summary['avg_stress_relative']:+.0%}")
    print(f"  Elevated states: {summary['elevated_pct']:.0f}%")
    
    # End session
    tracker.end_session()
    print(f"\n  Data saved to: {tracker.profile_dir}")
    
    print("\n" + "=" * 60)
    print("Calibration complete! System now tracks YOU, not averages.")
    print("=" * 60)


if __name__ == "__main__":
    quick_calibration_demo()
