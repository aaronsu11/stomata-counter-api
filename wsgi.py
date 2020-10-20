# -------------------------------------
# used by gunicorn to run the flask app
# -------------------------------------
from .manage import app

if __name__ == '__main__':
    app.run()
