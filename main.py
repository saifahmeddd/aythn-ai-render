from flask import Flask
from flask.blueprints import Blueprint
import config
import routes
from dotenv import load_dotenv
from flask_cors import CORS
from database.models import initialize_database

from config import FRONTEND_URL

# Load environment variables
load_dotenv()

# Initialize database
initialize_database()

# Initialize Flask app
app = Flask(__name__)


CORS(app, origins=[FRONTEND_URL])

# Enable debug mode
app.debug = config.DEBUG

# Register blueprints
for blueprint in vars(routes).values():
    if isinstance(blueprint, Blueprint):
        app.register_blueprint(blueprint, url_prefix=f'{blueprint.url_prefix}')

if __name__ == "__main__":
    print(f"Application running on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT)