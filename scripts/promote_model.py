# promote model: Script to promote a model from @development to @champion alias in the MLflow Model Registry

import os
import mlflow
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "aashu-0"
repo_name = "MLOps_Learning_Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


def get_model_version_by_alias(model_name, alias):
    """Get the model version associated with a specific alias.
    
    Args:
        model_name (str): Name of the registered model.
        alias (str): Alias name (e.g., '@development', '@champion').
        
    Returns:
        str: The version number of the model with the given alias.
        
    Raises:
        ValueError: If no model version found for the given alias.
    """
    client = mlflow.MlflowClient()
    try:
        registered_model = client.get_registered_model(model_name)
        # Check for the specified alias
        for model_alias in registered_model.aliases:
            if model_alias.alias.lower() == alias.lower():
                logging.info(f"Found model {model_name} with alias @{alias}: version {model_alias.version}")
                return model_alias.version
    except Exception as e:
        logging.error(f"Failed to retrieve model version for {model_name} with alias @{alias}: {e}")
        raise
    raise ValueError(f"Model {model_name} not found or has no version with alias @{alias}")


def promote_model_to_champion():
    """Promote model from @development alias to @champion alias."""
    client = mlflow.MlflowClient()
    model_name = "LR_Classifier"
    
    try:
        # Get the latest version with @development alias
        development_version = get_model_version_by_alias(model_name, "development")
        logging.info(f"Found development version: {development_version}")

        # Remove @champion alias from current champion model (if any)
        try:
            registered_model = client.get_registered_model(model_name)
            for model_alias in registered_model.aliases:
                if model_alias.alias.lower() == "champion":
                    logging.info(f"Removing @champion alias from version {model_alias.version}")
                    client.delete_registered_model_alias(model_name, "champion")
                    break
        except Exception as e:
            logging.info(f"No existing @champion alias to remove: {e}")

        # Set @champion alias to the development version
        client.set_registered_model_alias(model_name, "champion", development_version)
        logging.info(f"Model version {development_version} promoted to @champion alias")
        print(f"Model version {development_version} promoted to @champion alias")
        
        return development_version
        
    except Exception as e:
        logging.error(f"Failed to promote model: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        promote_model_to_champion()
    except Exception as e:
        logging.error(f"Failed to promote model: {e}")
        print(f"Error: {e}")
        raise
