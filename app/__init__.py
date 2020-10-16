# ------------------------------------------
# Define the current verion of the Flask API
# ------------------------------------------
from flask_restx import Api
from flask import Blueprint

from .main.controller.user_controller import api as user_ns
from .main.controller.image_controller import api as image_ns

# Blueprint allows mounting API on any url prefix (version) and/or subdomain
blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='STOMATA COUNTER FLASK RESTX API',
          version='1.0',
          description='a flask restx web api for processing stomata images'
          )

api.add_namespace(user_ns, path='/user')
api.add_namespace(image_ns, path='/image')
