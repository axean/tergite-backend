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


from flask_cors import CORS
from flask import jsonify, Blueprint, current_app, g, request
from flask import make_response
from werkzeug.local import LocalProxy

bcc_routes = Blueprint('bcc_routes','bcc_routes')
CORS(bcc_routes)

@bcc_routes.route('/bcc/registration', methods=['GET','POST'])
def api_registration():
    response = dict()

    if request.method == 'POST':
        if request.is_json:
            invitation = request.get_json()
            sha_check = invitation.get('sha_check',None)

            if sha_check:
                print('Got sha_check')
                response['name'] = "chalmers-gold"
                response['url']  = "https://qdp-git.mc2.chalmers.se/bcc"
                response['lab_setup'] = None
                response['description'] = "Stubbed BCC running on VM"
                response['sha_check'] = invitation['sha_check']
                response['status'] = {}
                response['config'] = {}
                response['calibration'] = []

                return make_response(jsonify(response), 200)
            else:
                response = 'No sha_check found'
                print(response)
                return make_response(jsonify(response), 400)

        else:
            response = "The POST payload was not a JSON!"
            print(response)
            return make_response(jsonify(response), 400)
    else:
        response="Send me an invitation!"
        print(response)
        return make_response(jsonify(response), 200)



