import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DEBUG = os.getenv("DEBUG_MODE", "True").lower() == "true"
HOST = os.getenv("HOST")
PORT = int(os.getenv("APPLICATION_PORT"))
FRONTEND_URL = os.getenv("FRONTEND_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_CONN_STRING = os.getenv("PG_CONN_STRING")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
META_APP_SECRET = os.getenv("META_APP_SECRET")
WEBHOOK_VERIFY_TOKEN = os.getenv("WEBHOOK_VERIFY_TOKEN")
META_APP_ID = os.getenv("META_APP_ID")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
WEBHOOK_CALLBACK_URL = os.getenv("WEBHOOK_CALLBACK_URL")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")

logging.basicConfig(
    filename=os.getenv("APP_LOG", "app.log"),
    level=logging.DEBUG,
    format="%(levelname)s: %(asctime)s \
        pid:%(process)s module:%(module)s %(message)s",
    datefmt="%d/%m/%y %H:%M:%S",
)
