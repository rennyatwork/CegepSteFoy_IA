from regression_model.config import config
import pandas as pd

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    print("Input Data:", input_data)
    print("Input Columns:", input_data.columns)

    validated_data = input_data.copy()

    # Check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        print("Numerical NA not allowed, handling...")
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    # Check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        print("Categorical NA not allowed, handling...")
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )

    # Check for values <= 0 for the log-transformed variables
    if (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().any():
        print("Log-transformed variables with values <= 0, handling...")
        vars_with_neg_values = config.NUMERICALS_LOG_VARS[
            (input_data[config.NUMERICALS_LOG_VARS] <= 0).any()
        ]
        validated_data = validated_data[validated_data[vars_with_neg_values] > 0]

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

    return validated_data
