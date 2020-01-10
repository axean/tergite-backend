# This code is part of Tergite
#
# (C) Copyright Miroslav Dobsicek 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from flask import Flask
from flask_cors import CORS
from bcc.api.routes import bcc_routes

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(bcc_routes)

    @app.route('/bcc')
    def serve():
        return "Welcome to dev' BCC"

    return app

