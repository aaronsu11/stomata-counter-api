# ------------------------------------------
# Define the current verion of the Flask API
# ------------------------------------------
from flask_restx import Api
from flask import Blueprint

from .main.controller.user_controller import api as user_ns

# Blueprint allows mounting API on any url prefix (version) and/or subdomain
blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='FLASK RESTPLUS API BOILER-PLATE WITH JWT',
          version='1.0',
          description='a boilerplate for flask restplus web service'
          )

api.add_namespace(user_ns, path='/user')