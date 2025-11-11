"""
Configuration settings for the application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")

# Application Settings
APP_NAME = "AgriSense XAI"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

# Model Settings
MODEL_DIR = Path(__file__).parent.parent / "modules" / "ml_engine" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "crop_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Data Settings
DATA_DIR = Path(__file__).parent.parent / "data"
LOCATIONS_FILE = DATA_DIR / "locations.json"

# ML Settings
TRAINING_SAMPLES = int(os.getenv("TRAINING_SAMPLES", "5000"))
MODEL_TRAINING_ENABLED = os.getenv("MODEL_TRAINING_ENABLED", "True").lower() == "true"

# XAI Settings
USE_SHAP = os.getenv("USE_SHAP", "True").lower() == "true"
USE_LIME = os.getenv("USE_LIME", "True").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

