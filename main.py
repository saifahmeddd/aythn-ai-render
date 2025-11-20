from flask import Flask
from flask.blueprints import Blueprint
import config
import routes
from dotenv import load_dotenv
from flask_cors import CORS
from database.models import initialize_database
import time
from config import FRONTEND_URL
from views.aythn import subscribe_webhook
import threading
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

def delayed_subscribe():
    time.sleep(5)
    try:
        result = subscribe_webhook()
        if result.get("error"):
            print(f"Webhook setup failed: {result}")
        else:
            print("Webhook setup completed")
            print(result)
    except Exception as e:
        print(f"Webhook subscribe exception: {e}")

if __name__ == "__main__":
    print(f"Application running on http://{config.HOST}:{config.PORT}")
    threading.Thread(target=delayed_subscribe).start()
    app.run(host=config.HOST, port=config.PORT)