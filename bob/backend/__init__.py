import os
from flask import Flask

def create_app():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    app = Flask(
        __name__,
        static_folder=os.path.join(project_root, "static"),
        template_folder=os.path.join(project_root, "templates")
    )
    app.secret_key = 'your_secret_key_here'  # 🔐 Replace with a secure key

    # Initialize global in-memory caches if desired
    app.image_cache = {}
    app.description_cache = {}
    app.user_guesses = {}

    # Load dataset
    from .load_data import load_dataset
    app.dataset = load_dataset(app)
    
    # Load model (and attach to app)
    from .load_model import load_model
    app.model = load_model(app)
    
    # Register routes
    from .routes import init_routes
    init_routes(app)
    
    return app
