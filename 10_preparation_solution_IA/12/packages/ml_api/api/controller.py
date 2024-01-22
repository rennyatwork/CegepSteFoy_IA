from flask import Blueprint, request, jsonify, render_template
from regression_model.predict import make_prediction
from regression_model import __version__ as _version

from api.config import get_logger
from api.validation import validate_inputs
from api import __version__ as api_version

import pandas as pd

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)

# Define the default route
@prediction_app.route('/', methods=['GET'])
def default_route():
    print("Inside default_route")
    #return render_template('index.html', template_folder='templates')  # Replace 'index.html' with the actual name of your HTML template
    return render_template('index.html')

@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})


@prediction_app.route('/v1/predict/regression', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using the provided validation function
        validated_data = validate_inputs(input_data=json_data)

        # If there are errors, return them
        if validated_data['errors']:
            return jsonify(validated_data), 400

        # Step 3: Model prediction
        result = make_prediction(input_data=validated_data['input_data'])
        _logger.debug(f'Outputs: {result}')

        # Step 4: Convert numpy ndarray to list
        predictions = result.get('predictions').tolist()
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': None})


@prediction_app.route('/test_validation', methods=['POST'])
def test_validation():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()
        _logger.debug(f'Inputs: {json_data}')

        # Step 2: Validate the input using the provided validation function
        validated_data = validate_inputs(input_data=json_data)

        return jsonify(validated_data)