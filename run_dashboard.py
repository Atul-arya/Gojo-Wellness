"""
Gojo Wellness Dashboard Server (v2.0)

A robust HTTP API server that connects the web dashboard to the Python
fractal analysis engine. Holds the PersonalTracker instance in memory
and saves data to disk permanently.

API Endpoints:
- GET  /api/status          -> Check calibration status
- POST /api/calibrate/add   -> Add calibration samples
- POST /api/calibrate/done  -> Finalize calibration
- POST /api/analyze         -> Analyze standardized mouse movement
- GET  /api/history         -> Get session history
"""

import http.server
import socketserver
import json
import os
import webbrowser
from urllib.parse import urlparse
from pathlib import Path
import sys

# Import the core engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personal_tracker import PersonalTracker

PORT = 8080
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")

# Global tracker instance
# Initializes with default profile and loads existing data from disk
tracker = PersonalTracker("default")


class DashboardAPI(http.server.SimpleHTTPRequestHandler):
    """API Handler for the dashboard."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)
    
    def _send_json(self, data, status=200):
        """Helper to send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
        
    def _read_json(self):
        """Helper to read JSON body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        return json.loads(self.rfile.read(content_length).decode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET endpoints."""
        parsed = urlparse(self.path)
        
        if parsed.path == '/api/status':
            # Return current tracker state
            data = {
                "is_calibrated": tracker.is_calibrated,
                "baseline": tracker.baseline.to_dict() if tracker.baseline else None,
                "calibration_samples": len(tracker.calibration_samples)
            }
            self._send_json(data)
            
        elif parsed.path == '/api/history':
            # Return today's history
            history = tracker.get_daily_history() + [s.to_dict() for s in tracker.session_history]
            # Deduplicate based on timestamp if needed, or largely just send recent
            # For dashboard, let's send last 100 entries for charts
            self._send_json(history[-100:])
            
        elif parsed.path == '/api/insight':
            # Get fractal insights
            result = tracker.get_insight()
            self._send_json(result)
            
        else:
            # Serve static files
            super().do_GET()
            
    def do_POST(self):
        """Handle POST endpoints."""
        parsed = urlparse(self.path)
        
        # Special handling for binary voice data
        if parsed.path == '/api/analyze/voice':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self._send_json({"error": "No audio data"}, status=400)
                    return
                    
                audio_data = self.rfile.read(content_length)
                result = tracker.analyze_voice(audio_data)
                self._send_json(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        # Standard JSON endpoints
        body = self._read_json()
        
        try:
            if parsed.path == '/api/calibrate/add':
                # Add sample to calibration
                result = tracker.add_calibration_sample(body['x'], body['y'])
                self._send_json(result)
                
            elif parsed.path == '/api/calibrate/done':
                # Finalize calibration
                baseline = tracker.finalize_calibration()
                self._send_json({
                    "status": "success", 
                    "baseline": baseline.to_dict()
                })
                
            elif parsed.path == '/api/analyze':
                # Run main analysis
                if not tracker.is_calibrated:
                    self._send_json({"error": "Not calibrated"}, status=400)
                    return
                
                snapshot = tracker.analyze_current(body['x'], body['y'])
                self._send_json(snapshot.to_dict())
                
            elif parsed.path == '/api/reset':
                # Reset everything
                tracker.reset_calibration()
                self._send_json({"status": "reset"})
                
            elif parsed.path == '/api/archive':
                # Manual trigger
                days = body.get('days', 7)
                tracker.archive_old_history(days_to_keep=days)
                self._send_json({"status": "archived"})
                
            elif parsed.path == '/api/tag':
                # Tag current state
                tag = body.get('tag')
                if tag:
                    tracker.tag_current_state(tag)
                    self._send_json({"status": "tagged", "tag": tag})
                else:
                    self._send_json({"error": "No tag provided"}, status=400)
                
            else:
                self.send_error(404, "Endpoint not found")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)


def run_server():
    """Start the dashboard server."""
    print("=" * 60)
    print("GOJO WELLNESS DASHBOARD v2.0")
    print("=" * 60)
    print(f"Backend active: Connecting to {os.path.abspath('personal_tracker.py')}")
    print(f"Data storage: {os.path.abspath('gojo_data')}")
    print(f"Server URL: http://localhost:{PORT}")
    print("\nPress Ctrl+C to stop.\n")
    
    # Auto-archive old history on startup
    print("Running maintenance: Archiving old history...")
    try:
        res = tracker.archive_old_history(days_to_keep=7)
        print(f"  Archived {res['files_archived']} files, cleaned {res['points_removed']} points.")
    except Exception as e:
        print(f"  Archive error: {e}")
    
    # Open browser
    webbrowser.open(f"http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), DashboardAPI) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            tracker.end_session()  # Ensure data is saved


if __name__ == "__main__":
    run_server()
