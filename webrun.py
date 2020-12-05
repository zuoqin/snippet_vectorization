from flask import Flask, jsonify, request
from flasgger import Swagger
from flask_jwt_extended import (create_access_token,
    create_refresh_token, jwt_required, jwt_refresh_token_required,
    get_jwt_identity, get_raw_jwt)
from datetime import datetime, timedelta
app = Flask(__name__)
Swagger(app)
from vocabularies import VocabType
from config import Config
from zuoqin_predict import ZuoqinPredictor
from model_base import CodeVectorizeModelBase


def load_main_model(config: Config) -> CodeVectorizeModelBase:
    assert config.DL_FRAMEWORK in {'tensorflow', 'keras'}
    if config.DL_FRAMEWORK == 'tensorflow':
        from tensorflow_model import CodeVectorizeModel
    elif config.DL_FRAMEWORK == 'keras':
        from keras_model import CodeVectorizeModel
    return CodeVectorizeModel(config)


from flask_jwt_extended import JWTManager
app.config['JWT_SECRET_KEY'] = 'jwt-secret-string'
#app.config['JWT_EXPIRATION_DELTA'] = timedelta(days=10)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=10)
jwt = JWTManager(app)

from flask_restful import reqparse, request, abort, Api, Resource
from flask_cors import CORS


CORS(app)
config = Config(set_defaults=True, load_from_args=True, verify=True)

model = load_main_model(config)
config.log('Done creating code vectorize model')
predictor = ZuoqinPredictor(config, model)

@app.route('/api/login', methods=['POST'])
def get_token():
    """
    This is Code As Vector API
    Call this method to authenticate
    ---
    tags:
      - Code As Vector API
    parameters:
      - in: body
        name: user
        schema:
            type: object
            required:
                - username
            properties:
                username:
                    type: string
                password:
                    type: string
    responses:
      500:
        description: Error
      200:
        description: Token to access resources
        schema:
          id: login_params
          properties:
            token:
              type: string
              description: Access code
              default: ''
    """
    parser = reqparse.RequestParser()
    parser.add_argument('username', help = 'This field cannot be blank', required = True)
    parser.add_argument('password', help = 'This field cannot be blank', required = True)
    data = parser.parse_args()

    if (data.username == 'tinkoff' and data.password == '5^j38W_GtZ') or \
       (data.username == 'andrey' and data.password == '5^j38W_GtZ'):
        access_token = create_access_token(identity = data['username'])
    else:
        access_token = ''
    result = {'token': access_token}
    return jsonify(
        result
    )


@app.route('/api/getresult', methods=['POST'])
@jwt_required
def get_result():
    """
    This is Code As Vector API
    Call this method and paste the function signature and code in body
    ---
    tags:
      - Code As Vector API
    parameters:
      - in: header
        name: Authorization
        type: string
        required: true
      - name: data
        in: body
        type: string
        required: true
        description: Function code to analyze
    responses:
      500:
        description: Error
      200:
        description: Function names
        schema:
          id: result_schema
          properties:
            names:
              type: string
              description: function names
              default: 'get'
    """

    #address = request.aargs.get('address', default = '*', type = str)
    #print(request.data)
    result = predictor.predict(request.data, False)
    #result = {'result': 'OK'}
    return jsonify(
        result
    )

@app.route('/api/getcodelabel', methods=['POST'])
def get_codelabel():
    """
    This is Code As Vector API
    Call this method and paste only function code without signature in body
    ---
    tags:
      - Code As Vector API
    parameters:
      - in: header
        name: Authorization
        type: string
        required: true
      - name: data
        in: body
        type: string
        required: true
        description: Function code to analyze
    responses:
      500:
        description: Error
      200:
        description: Function names
        schema:
          id: result_schema
          properties:
            names:
              type: string
              description: function names
              default: 'get'
    """

    #address = request.aargs.get('address', default = '*', type = str)
    #print(request.data)
    result = predictor.predict(request.data, True)
    #result = {'result': 'OK'}
    return jsonify(
        result
    )


@app.route('/api/classify_pattern', methods=['POST'])
def classify_pattern():
    """
    This is Code As Vector API
    Call this method and paste java code with two functions separated with -- delimiter  in body
    ---
    tags:
      - Code As Vector API
    parameters:
      - in: header
        name: Authorization
        type: string
        required: true
      - name: data
        in: body
        type: string
        required: true
        description: Two Functions delimited with -- separator which contain original and target pattern codes
    responses:
      500:
        description: Error
      200:
        description: Classificator and probability
        schema:
          id: result_schema
          properties:
            names:
              type: string
              description: function names
              default: 'get'
    """
    from patterns import get_result
    result = get_result(request.data.decode("utf-8"))
    result = {'classificator': int(result[0][0]), 'probability': result[1][0][result[0][0]]}
    return jsonify(
        result
    )

app.run(debug=True,host='0.0.0.0', port=8080)
