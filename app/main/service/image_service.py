# -----------------------------------------------------------------
# This class handles all the logic relating to the image processing
# -----------------------------------------------------------------
import boto3
from flask import current_app
from ..core import MRCNNStomataDetector

# defining s3 bucket object
s3 = boto3.client("s3")

detector = MRCNNStomataDetector()


def process_image(url):
    # list all file in bucket
    bucket_name = current_app.config['S3_BUCKET_NAME']
    # fetching object from bucket
    file_obj = s3.get_object(Bucket=bucket_name, Key="stomata1.jpg")
    # reading the file content in bytes
    image_bytes = file_obj["Body"].read()

    image_bytes, num_stomata, scores, areas = detector.process_image(
        image_bytes)

    s3.put_object(Bucket=bucket_name,
                  Key="stomata_result.jpg", Body=image_bytes)

    presigned_url = s3.generate_presigned_url('get_object', Params={
                                              'Bucket': bucket_name, 'Key': "stomata_result.jpg"}, ExpiresIn=3600)

    response_object = {
        'presigned_url': presigned_url,
        'num_stomata': num_stomata,
        'scores': scores,
        'areas': areas
    }

    return response_object, 201


def process_image_batch(data):
    user = None
    if not user:
        response_object = {
            'status': 'success',
            'message': 'Successfully registered.'
        }
        return response_object, 201
    else:
        response_object = {
            'status': 'fail',
            'message': 'User already exists. Please Log in.',
        }
        return response_object, 409


def list_images():
    bucket_name = current_app.config['S3_BUCKET_NAME']
    # list all file in bucket
    response = s3.list_objects_v2(
        Bucket=bucket_name
    )
    objs = []
    for obj in response.get('Contents', None):
        objs.append(obj.get('Key', None))
    response_object = {
        'status': 'success',
        'objects': objs,
    }
    return response_object, 200
