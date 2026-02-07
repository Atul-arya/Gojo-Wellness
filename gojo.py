"""
GOJO - Personal Wellness Commander (CLI)

The central command line interface for your Personal Wellness Tracker.
Controls the background daemon, generates reports, and handles manual check-ins.

Commands:
  start   - Launch background tracking daemon (silent)
  stop    - Stop the daemon
  report  - Show today's wellness summary
  voice   - Record morning voice check-in (10s)
  status  - Check if tracker is running
"""

import sys
import os
import argparse
import subprocess
import json
import time
from datetime import datetime
import signal
import numpy as np

# Add local path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personal_tracker import PersonalTracker

DAEMON_SCRIPT = "background_daemon.py"
PID_FILE = "daemon.pid"

def cmd_start():
    """Start the background daemon."""
    if os.path.exists(PID_FILE):
        print("Daemon might already be running (check daemon.pid).")
        print("Run 'gojo stop' first if you want to restart.")
        return

    print("Starting background tracker...")
    if sys.platform == 'win32':
        # Windows: Use DETACHED_PROCESS for background execution
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen([sys.executable, DAEMON_SCRIPT], 
                         creationflags=DETACHED_PROCESS,
                         close_fds=True)
    else:
        subprocess.Popen([sys.executable, DAEMON_SCRIPT], 
                         start_new_session=True)
        
    print("Daemon started! It will track your stress silently.")
    print("Run 'gojo status' to verify.")

def cmd_stop():
    """Stop the background daemon."""
    if not os.path.exists(PID_FILE):
        print("No daemon.pid found. Is it running?")
        return
        
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            
        print(f"Stopping daemon (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
        
        # Clean up
        os.remove(PID_FILE)
        print("Stopped.")
        
    except Exception as e:
        print(f"Error stopping daemon: {e}")
        try:
            os.remove(PID_FILE) # Force clean
        except: pass

def cmd_status():
    """Check status."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            pid = f.read().strip()
        print(f"[+] Daemon is RUNNING (PID {pid})")
    else:
        print("[-] Daemon is STOPPED")
        
    tracker = PersonalTracker("default")
    if tracker.is_calibrated:
        print(f"[*] Baseline: Learned (H={tracker.baseline.h_mean:.2f})")
    else:
        print("[!] Not Calibrated. Run 'gojo start' and use your PC for a while.")

def cmd_report():
    """Generate daily report."""
    tracker = PersonalTracker("default")
    today = datetime.now().strftime("%Y-%m-%d")
    history = tracker.get_daily_history(today)
    
    if not history:
        print(f"No data found for today ({today}).")
        return
        
    print("\n" + "="*40)
    print(f"DAILY REPORT: {today}")
    print("="*40)
    
    # Calculate stats
    total_snapshots = len(history)
    hours_logged = (total_snapshots * 15) / 3600 # Approx 15s per snapshot in daemon
    
    # Stress
    stress_vals = [h.get('stress_relative', 0) for h in history]
    avg_stress = sum(stress_vals) / len(stress_vals)
    peak_stress = max(stress_vals)
    
    # States
    states = [h.get('state', 'unknown') for h in history]
    from collections import Counter
    if states:
        common_state = Counter(states).most_common(1)[0][0]
    else:
        common_state = "unknown"
    
    print(f"[#]  Active Time: {hours_logged:.1f} hours")
    print(f"[*]  Primary State: {common_state.upper()}")
    
    # Stress Bar
    bar_len = 20
    filled = int((avg_stress + 1) / 2 * bar_len) # Map -1..1 to 0..20
    filled = max(0, min(bar_len, filled))
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"[-]  Stress Level:  [{bar}] {avg_stress:+.2f}")
    
    if peak_stress > 0.8:
        print("\n[!]  WARNING: High burnout risk detected today.")
    
    print("="*40 + "\n")

def cmd_voice():
    """Record morning check-in."""
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
    except ImportError:
        print("Error: Missing libraries.")
        print("Please run: pip install sounddevice scipy")
        return

    print("\n[*] Start speaking in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("[*] RECORDING (10s) - Speak naturally...")
    
    fs = 44100
    seconds = 10
    
    try:
        # sounddevice returns float32
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        print("Processing...")
        
        # Convert to int16 (PCM) because 'wave' module doesn't like float
        recording_int16 = (recording * 32767).astype(np.int16)
        
        # Save temp
        temp_file = "temp_voice.wav"
        wav.write(temp_file, fs, recording_int16)
        
        # Analyze
        with open(temp_file, 'rb') as f:
            audio_data = f.read()
            
        tracker = PersonalTracker("default")
        result = tracker.analyze_voice(audio_data)
        
        if result.get("error"):
            print(f"[!] Analysis Failed: {result['error']}")
        else:
            print("\n" + "-"*30)
            print("VOICE INSIGHTS")
            print("-"*30)
            print(f"State: {result['state'].upper()}")
            print(f"Energy: {result['energy']*100:.0f}%")
            print(f"Stress: {result['stress']*100:.0f}%")
            
            if result.get('alerts'):
                print(f"Alerts: {', '.join(result['alerts'])}")
                
        os.remove(temp_file)
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Gojo Wellness CLI")
    parser.add_argument("command", choices=["start", "stop", "report", "voice", "status"])
    
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    
    if args.command == "start":
        cmd_start()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "status":
        cmd_status()
    elif args.command == "report":
        cmd_report()
    elif args.command == "voice":
        cmd_voice()

if __name__ == "__main__":
    main()
