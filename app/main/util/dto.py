# ----------------------------------------------------------------------------------------
# This data transfer object (DTO) will be responsible for carrying data between processes.
# ----------------------------------------------------------------------------------------

from flask_restx import Namespace, fields


class UserDto:
    api = Namespace('user', description='user related operations')
    user = api.model('user', {
        'email': fields.String(required=True, description='user email address'),
        'username': fields.String(required=True, description='user username'),
        'password': fields.String(required=True, description='user password'),
        'public_id': fields.String(description='user Identifier')
    })


class ImageDto:
    api = Namespace('image', description='image related operations')
    image = api.model('image', {
        'name': fields.String(required=True, description='image name'),
        'url': fields.String(required=True, description='image URL'),
    })
    image_data = api.model('image_data', {
        'num_stomata': fields.Integer(required=True, description='number of stomata detected'),
        'bbox': fields.List(fields.Float),
    })
