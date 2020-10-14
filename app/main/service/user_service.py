# ---------------------------------------------------------------
# This class handles all the logic relating to the user database.
# ---------------------------------------------------------------

import uuid
import datetime

def save_new_user(data):
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


def get_all_users():
    return True


def get_a_user(public_id):
    return True


def save_changes(data):
    pass
