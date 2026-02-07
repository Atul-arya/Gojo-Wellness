"""
Background Daemon for Personal Wellness Tracker (Headless Mode)

This script runs silently in the background, tracking mouse movements
and analyzing them for stress/burnout patterns every 15 seconds.

It uses `ctypes` for low-level hook-free polling, which is lightweight.

Alerts:
- System Beep (winsound) if stress > 0.8
- Logs to gojo_data/history_YYYY-MM-DD.json

Usage:
    python background_daemon.py
"""

import time
import ctypes
import winsound
import sys
import os
from datetime import datetime
from typing import List, Tuple

# Ensure we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personal_tracker import PersonalTracker, StateSnapshot

# Windows API for mouse position
class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def get_cursor_pos() -> Tuple[int, int]:
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)

def run_daemon():
    # Redirect output to log file
    sys.stdout = open("daemon.log", "a")
    sys.stderr = sys.stdout
    
    pid = os.getpid()
    print(f"\n[{datetime.now()}] Background Daemon Started. PID: {pid}")
    
    # Save PID for stopper
    with open("daemon.pid", "w") as f:
        f.write(str(pid))
        
    print("Tracking mouse activity... (Press Ctrl+C to stop manually)")
    
    tracker = PersonalTracker("default")
    buffer_x = []
    buffer_y = []
    last_analysis = time.time()
    
    try:
        while True:
            # Poll every 50ms (20Hz)
            x, y = get_cursor_pos()
            buffer_x.append(x)
            buffer_y.append(y)
            time.sleep(0.05)
            
            # Analyze every 15 seconds
            if time.time() - last_analysis > 15:
                # Need enough data points
                if len(buffer_x) > 100:
                    try:
                        # Analyze
                        if tracker.is_calibrated:
                            snapshot = tracker.analyze_current(buffer_x, buffer_y)
                            
                            # Check Alerts
                            if snapshot.stress_relative > 0.6:
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] HIGH STRESS: {snapshot.stress_relative:.2f}")
                                # Gentle beep for warning
                                winsound.Beep(440, 200) 
                                if snapshot.stress_relative > 0.8:
                                    winsound.Beep(440, 500) # Longer beep for critical
                                    
                            # Check Insight Engine
                            insight = tracker.get_insight()
                            if insight.get('alert'):
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] PREDICITION: {insight['alert']}")
                                winsound.Beep(880, 200)
                                winsound.Beep(880, 200)
                        
                        # Force save for immediate visibility
                        tracker._save_history()

                    except Exception as e:
                        print(f"Analysis Error: {e}")
                
                # Reset buffer (sliding window overlap could be better, but simple flush is fine for daemon)
                # Actually, BioAdapter likes continuity, but for "current state" snapshots, flush is okay
                # providing we have enough points.
                buffer_x = []
                buffer_y = []
                last_analysis = time.time()
                
                # Auto-archive check periodically (every 100 analysis cycles? approx 25 mins)
                # Or just let the tracker handle saving history (it autosaves every 50 snapshots)
                
    except KeyboardInterrupt:
        print("Stopping daemon...")
        tracker.end_session()

if __name__ == "__main__":
    run_daemon()
