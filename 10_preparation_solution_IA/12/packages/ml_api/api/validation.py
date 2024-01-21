from regression_model.config import config
import pandas as pd

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    print("Input Data:", input_data)
    print("Input Columns:", input_data.columns)

    # Check for any additional columns present in the input data
    extra_columns = set(input_data.columns) - set(config.FEATURES)
    if extra_columns:
        print(f"Unexpected columns in input data: {', '.join(extra_columns)}")
        raise ValueError(f"Unexpected columns in input data: {', '.join(extra_columns)}")

    # Check for any missing columns expected by the model
    missing_columns = set(config.FEATURES) - set(input_data.columns)
    if missing_columns:
        print(f"Missing columns in input data: {', '.join(missing_columns)}")
        raise ValueError(f"Missing columns in input data: {', '.join(missing_columns)}")

    # Additional checks or modifications can be added here based on your requirements

    return input_data