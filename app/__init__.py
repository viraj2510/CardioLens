from flask import Flask

def create_app():
    # Create and configure the app
    app = Flask(__name__)
    
    # Register the routes from the routes.py file
    with app.app_context():
        from . import routes
        
    return app