"""
Application Runner
Starts the Flask Web Application for Election Social Media Sentiment Mining.
"""

import sys
import argparse
from app import create_app

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Election Sentiment Mining Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    print(f"\n==================================================================")
    print(f" VotePulse AI: Election Social Media Sentiment Analysis via Text Mining")
    print(f" Web Server running at: http://{args.host}:{args.port}/")
    print(f" Press Ctrl+C to terminate.")
    print(f"==================================================================\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
