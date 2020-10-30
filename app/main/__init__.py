# --------------------------------------------
# Instantiate the Flask app by a configuration
# --------------------------------------------
from flask import Flask
from flask_cors import CORS

from .config import config_by_name

cors = CORS()


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    cors.init_app(app)

    return app
