# ---------------------------------------------------------------------------------------
# The user controller class handles all the incoming HTTP requests relating to the image.
# ---------------------------------------------------------------------------------------

from flask import request
from flask_restx import Resource

from ..util.dto import ImageDto
from ..service.image_service import process_image, process_image_batch, list_images

# using the "image" namespace
api = ImageDto.api
_image = ImageDto.image
_image_data = ImageDto.image_data


@api.route('/detect')
class Image(Resource):
    @api.doc('get the processed image')
    @api.marshal_list_with(_image, envelope='image')
    def get(self):
        """Get the URL of processed image"""
        return "s3://ainz11-test/..."

    @api.response(201, 'Image successfully processed.', _image_data)
    @api.doc('process an image given its url')
    @api.expect(_image, validate=True)
    def post(self):
        """Process an image given the URL from S3"""
        meta = request.json
        image_name = meta["name"]
        return process_image(image_name)


@api.route('/list')
class ImageList(Resource):
    @api.doc('DEV - get the list of images in the default bucket')
    def get(self):
        """List all registered users"""
        return list_images()
