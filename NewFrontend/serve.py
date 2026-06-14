"""Start a local HTTP server for the QuarryFlow visualizer.

Usage:
    python docs/visualizer/serve.py

Opens the visualizer in your default browser with full data loading support.
"""
import http.server
import os
import sys
import threading
import webbrowser
from functools import partial
from pathlib import Path

PORT = 8000
DIR = Path(__file__).resolve().parent


def main():
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DIR))
    with http.server.HTTPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/"
        print(f"Serving visualizer at {url}")
        print("Press Ctrl+C to stop.")
        # Open browser after a short delay
        threading.Timer(0.5, webbrowser.open, args=[url]).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
