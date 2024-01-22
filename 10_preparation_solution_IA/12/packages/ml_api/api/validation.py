from regression_model.config import config
import pandas as pd

# def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
#     """Check model inputs for unprocessable values."""

#     print("Input Data:", input_data)
#     print("Input Columns:", input_data.columns)

#     # Check for any additional columns present in the input data
#     extra_columns = set(input_data.columns) - set(config.FEATURES)
#     if extra_columns:
#         print(f"Unexpected columns in input data: {', '.join(extra_columns)}")
#         raise ValueError(f"Unexpected columns in input data: {', '.join(extra_columns)}")

#     # Check for any missing columns expected by the model
#     missing_columns = set(config.FEATURES) - set(input_data.columns)
#     if missing_columns:
#         print(f"Missing columns in input data: {', '.join(missing_columns)}")
#         raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

#     # Additional checks or modifications can be added here based on your requirements

#     return input_data

from regression_model.config import config
import pandas as pd

def validate_inputs(input_data: dict) -> dict:
    """Check model inputs for unprocessable values."""

    # Extract 'data' key
    data = input_data.get('data', [])

    # If 'data' is a list of dictionaries, convert it to a DataFrame
    if isinstance(data, list) and data:
        input_df = pd.DataFrame(data)
    else:
        # If 'data' is not a list of dictionaries, assume it's already a DataFrame
        input_df = pd.DataFrame(data) if isinstance(data, dict) else pd.DataFrame()

    # Check for any additional columns present in the input data
    extra_columns = set(input_df.columns) - set(config.FEATURES)

    # Include extra columns in errors and return the input_data
    if extra_columns:
        errors = f"Unexpected columns in input data: {', '.join(extra_columns)}"
        print(errors)
        return {'input_data': input_df.to_dict(orient='records'), 'errors': errors}

    # Check for any missing columns expected by the model
    missing_columns = set(config.FEATURES) - set(input_df.columns)

    # Include missing columns in errors and return the input_data
    if missing_columns:
        errors = f"Missing columns in input data: {', '.join(missing_columns)}"
        print(errors)
        return {'input_data': input_df.to_dict(orient='records'), 'errors': errors}

    return {'input_data': input_df.to_dict(orient='records'), 'errors': None}
