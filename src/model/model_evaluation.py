# model_evaluation: Script to evaluate a trained Logistic Regression model using test data and log metrics to MLflow.

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.logger import logging


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DASGHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DASGHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "aashu-0"
repo_name = "MLOps_Learning_Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/aashu-0/MLOps_Learning_Project.mlflow')
# dagshub.init(repo_owner='aashu-0', repo_name='MLOps_Learning_Project', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)

        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("LR_Model_Eval") 
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./dataset/processed/test_tfidf.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log model to MLflow using the name parameter (modern approach)
            model_info = mlflow.sklearn.log_model(sk_model=clf, artifact_path="model")
            logging.info(f'Model artifact path: {model_info.artifact_path}')
            logging.info(f'Model artifact URI: {model_info.model_uri}')
            
            # Log metrics to MLflow
            try:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                logging.info('Metrics logged to MLflow')
            except Exception as e:
                logging.warning('Could not log metrics to MLflow: %s. Continuing...', e)
            
            # Log model parameters to MLflow
            try:
                if hasattr(clf, 'get_params'):
                    params = clf.get_params()
                    for param_name, param_value in params.items():
                        mlflow.log_param(param_name, param_value)
                    logging.info('Model parameters logged to MLflow')
            except Exception as e:
                logging.warning('Could not log parameters to MLflow: %s. Continuing...', e)
            
            # Log the metrics file to MLflow
            try:
                mlflow.log_artifact('reports/metrics.json')
                logging.info('Metrics file artifact logged to MLflow')
            except Exception as e:
                logging.warning('Could not log metrics artifact to MLflow: %s. Continuing...', e)
            
            # Save model info - artifact path should match the one used in log_model
            model_info_dict = {'run_id': run.info.run_id, 'model_uri': model_info.model_uri}

            with open('reports/experiment_info.json', 'w') as f:
                json.dump(model_info_dict, f, indent=4)
            logging.info('Model info saved with correct model URI')
            logging.info(f'Model logged successfully with run ID: {run.info.run_id}')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
