# register model: This script registers a trained model to the MLflow Model Registry

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"

# Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/aashu-0/MLOps_Learning_Project.mlflow')
dagshub.init(repo_owner='aashu-0', repo_name='MLOps_Learning_Project', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        # Use the model_uri directly from the model_info
        model_uri = model_info.get('model_uri')
        client = mlflow.tracking.MlflowClient()
        
        # Try to register the model; if it already exists, create a new version
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(f'Model {model_name} registered successfully')
        
        # Set the "staging" alias for the registered model version
        client.set_registered_model_alias(model_name, "staging", model_version.version)
        logging.info(f'Model {model_name} version {model_version.version} registered and aliased as "staging".')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "LR_Classifier"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()