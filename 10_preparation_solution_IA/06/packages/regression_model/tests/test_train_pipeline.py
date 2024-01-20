# tests/test_train_pipeline.py
import pytest
from unittest.mock import patch
from regression_model.train_pipeline import run_training, logging

def test_run_training():
    ## given
    with patch("regression_model.train_pipeline.load_dataset") as mock_load_dataset:
        test_data = {}  # Adjust this based on your actual test data

        # Mock save_pipeline to avoid actually saving the pipeline during tests
        with patch("regression_model.train_pipeline.save_pipeline") as mock_save_pipeline:
            ## when
            run_training()

            ## then
            # Add assertions to check the expected behavior

            # Check if load_dataset is called with the correct file_name
            assert mock_load_dataset.call_count == 1
            assert mock_load_dataset.call_args[1]['file_name'] == 'test.csv'

            # Check if save_pipeline is called
            mock_save_pipeline.assert_called_once()

            # Check if the logger is called with the correct message
            assert any("saving model version" in log[0] for log in logging._messages["regression_model"])

            # Add more assertions based on your specific use case
