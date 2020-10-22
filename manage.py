# ----------------------------------------
# Manage how the app is run, i.e. serve/test
# ----------------------------------------
import os
import unittest

from flask_script import Manager

from app import blueprint
from app.main import create_app

app = create_app(os.getenv('BOILERPLATE_ENV') or 'dev')

# calling Api.init_app() is not required here because
# registering the blueprint with the app takes care of
# setting up the routing for the application.
app.register_blueprint(blueprint)

app.app_context().push()

manager = Manager(app)


@manager.command
def run():
    app.run()


@manager.command
def test():
    """Runs the unit tests."""
    tests = unittest.TestLoader().discover('app/test', pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if result.wasSuccessful():
        return 0
    return 1


if __name__ == '__main__':
    manager.run()
