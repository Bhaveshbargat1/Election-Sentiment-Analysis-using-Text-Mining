"""
Flask Application Factory
Initializes the Flask app, configurations, and registers blueprints/routes.
"""

import os
from flask import Flask

def create_app():
    app = Flask(__name__, 
                template_folder="templates",
                static_folder="static")
    
    app.config["SECRET_KEY"] = "election-sentiment-textmining-2026"
    
    # Base paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    app.config["BASE_DIR"] = base_dir
    app.config["DATA_PATH"] = os.path.join(base_dir, "data", "election_tweets_processed.csv")
    app.config["MODELS_DIR"] = os.path.join(base_dir, "models")
    app.config["STATIC_IMG_DIR"] = os.path.join(base_dir, "app", "static", "images")
    
    with app.app_context():
        from app import routes
        app.register_blueprint(routes.bp)
        
    return app
