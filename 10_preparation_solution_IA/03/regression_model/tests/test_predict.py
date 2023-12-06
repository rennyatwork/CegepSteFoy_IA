import sys
import os

# Add the path to the project's root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import math

from regression_model.predict import make_prediction
from regression_model.processing.data_management import load_dataset


import sys
print(sys.path)

import os
print(os.environ.get('PYTHONPATH'))


def test_make_single_prediction():
    # Given
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert math.ceil(subject.get('predictions')[0]) == 112476
