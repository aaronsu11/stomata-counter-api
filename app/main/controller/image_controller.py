# ---------------------------------------------------------------------------------------
# The user controller class handles all the incoming HTTP requests relating to the image.
# ---------------------------------------------------------------------------------------

from flask import request
from flask_restx import Resource

from ..util.dto import ImageDto
from ..service.image_service import MRCNNStomataDetector

# using the "image" namespace
api = ImageDto.api
_image = ImageDto.image
_image_data = ImageDto.image_data


detector = MRCNNStomataDetector()


def dummy(data='google.com'):
    return 'google.com'


@api.route('/')
class Image(Resource):
    @api.doc('get the processed image')
    @api.marshal_list_with(_image, envelope='image')
    def get(self):
        """Get the URL of processed image"""
        return dummy()

    @api.response(201, 'Image successfully processed.', _image_data)
    @api.doc('process an image given its url')
    @api.expect(_image, validate=True)
    def post(self):
        """Process an image given the URL from S3"""
        data = request.json
        return detector.process_image("dummy")
