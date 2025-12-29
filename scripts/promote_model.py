# promote model: Script to promote a model from @Candidate to @Champion alias in the MLflow Model Registry

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

def promote_model_to_champion():
    """Promote model from @Candidate alias to @Champion alias.
    
    Skips promotion if @Champion already points to the same version as @Candidate.
    
    Raises:
        ValueError: If @Candidate alias is not found
        RuntimeError: If model retrieval or promotion fails
    """
    client = mlflow.MlflowClient()
    registered_model_name = "LR_Classifier"
    
    # Retrieve registered model
    try:
        registered_model = client.get_registered_model(registered_model_name)
        logging.info(f"Retrieved registered model: {registered_model_name}")
    except Exception as e:
        error_msg = f"Failed to get registered model {registered_model_name}: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Check if @Candidate alias exists
    if 'Candidate' not in registered_model.aliases:
        error_msg = f"No @Candidate alias found for {registered_model_name}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    candidate_version = registered_model.aliases['Candidate']
    logging.info(f"Found @Candidate alias pointing to version {candidate_version}")

    # Check if @Champion alias already exists and points to the same version
    champion_version = registered_model.aliases.get('Champion')
    
    if champion_version and str(champion_version) == candidate_version:
        logging.info(f"Model version {candidate_version} is already the @Champion. No promotion needed.")
        print(f"Model version {candidate_version} is already the @Champion")
        return
    
    # Remove old @Champion alias if it exists on a different version
    if champion_version:
        logging.info(f"Removing @Champion alias from version {champion_version}")
        try:
            client.delete_registered_model_alias(registered_model_name, "Champion")
            logging.info(f"Successfully removed @Champion alias from version {champion_version}")
        except Exception as e:
            logging.warning(f"Failed to remove old @Champion alias: {e}")

    # Promote candidate to champion
    try:
        client.set_registered_model_alias(registered_model_name, "Champion", candidate_version)
        logging.info(f"Model version {candidate_version} promoted to @Champion alias")
        print(f"Model version {candidate_version} promoted to @Champion alias")
    except Exception as e:
        error_msg = f"Failed to promote model version {candidate_version} to @Champion: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        promote_model_to_champion()
    except Exception as e:
        logging.error(f"Failed to promote model: {e}")
        print(f"Error: {e}")
        raise
