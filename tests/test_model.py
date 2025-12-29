# test_model: Tests for loading and validating the ML model from MLflow registry

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
from dotenv import load_dotenv

load_dotenv()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Helper function to preprocess a single text string."""
    if not isinstance(text, str):
        return ""

    #1. Lowercase & remove URLs
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    #2. Remove numbers & punctuations
    text = re.sub(r'\d+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")

    #3. tokenize, stop word removal & lemmatization
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words).strip()

class TestChampionModel(unittest.TestCase):
    """Test suite for the @Champion aliased model"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the Champion model."""
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

        # Load the model from MLflow model registry using @Champion alias
        cls.model_name = "LR_Classifier"
        cls.alias = "Champion"
        cls.model_uri = f'models:/{cls.model_name}@{cls.alias}'
        
        try:
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
            logging.info(f"Successfully loaded model {cls.model_name} with @{cls.alias} alias")
        except Exception as e:
            error_msg = f"Failed to load champion model with @{cls.alias} alias: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Load the vectorizer
        vectorizer_path = 'models/vectorizer.pkl'
        try:
            with open(vectorizer_path, 'rb') as f:
                cls.vectorizer = pickle.load(f)
            logging.info(f"Loaded vectorizer from {vectorizer_path}")
        except FileNotFoundError as e:
            error_msg = f"Vectorizer not found at {vectorizer_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load vectorizer: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Load holdout test data
        test_data_path = 'dataset/processed/test_tfidf.csv'
        try:
            cls.holdout_data = pd.read_csv(test_data_path)
            logging.info(f"Loaded test data from {test_data_path} with shape {cls.holdout_data.shape}")
        except FileNotFoundError as e:
            error_msg = f"Holdout test data not found at {test_data_path}"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load test data: {e}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e

    def test_champion_alias_exists(self):
        """Test that the @Champion alias exists for the model."""
        client = mlflow.MlflowClient()
        try:
            registered_model = client.get_registered_model(self.model_name)
            alias_dict = registered_model.aliases
            self.assertIn(self.alias, alias_dict, f"@{self.alias} alias not found for model {self.model_name}")
            champion_version = alias_dict[self.alias]
            logging.info(f"@{self.alias} alias exists for {self.model_name}, pointing to version {champion_version}")
        except Exception as e:
            error_msg = f"Failed to verify @{self.alias} alias: {e}"
            logging.error(error_msg)
            self.fail(error_msg)

    def test_model_can_be_loaded_with_alias(self):
        """Test that the champion model can be loaded successfully via alias."""
        try:
            model = mlflow.pyfunc.load_model(self.model_uri)
            self.assertIsNotNone(model, f"Model should load successfully with @{self.alias} alias")
            logging.info(f"Successfully loaded {self.model_name} with @{self.alias} alias")
        except Exception as e:
            error_msg = f"Failed to load model with @{self.alias} alias: {e}"
            logging.error(error_msg)
            self.fail(error_msg)

    def test_model_input_signature(self):
        """Test model accepts vectorized input and produces valid predictions."""
        try:
            # dummy input text
            input_text = "hi how are you"
            processed_text = preprocess_text(input_text)
            input_data = self.vectorizer.transform([processed_text])

            feature_names = self.vectorizer.get_feature_names_out()
            input_df = pd.DataFrame(input_data.toarray())

            # make prediction
            prediction = self.model.predict(input_df)

            # check number of columns in input matches vectorizer features length
            self.assertEqual(input_df.shape[1], len(feature_names), 
                            "Input columns should match vectorizer feature length")

            # check prediction output shape
            self.assertEqual(len(prediction), input_df.shape[0], 
                            "Prediction should have one output per input row")
            self.assertEqual(len(prediction.shape), 1, 
                            "Prediction should be 1-dimensional")
            logging.info(f"Model input signature validated - features: {len(feature_names)}, predictions: {len(prediction)}")
        except Exception as e:
            error_msg = f"Failed to validate model input signature: {e}"
            logging.error(error_msg)
            self.fail(error_msg)

    def test_model_prediction_output_type(self):
        """Test that model predictions are valid (0 or 1 for binary classification)."""
        try:
            X_holdout = self.holdout_data.iloc[:, 0:-1]
            y_pred = self.model.predict(X_holdout)
            
            # Check predictions are binary (0 or 1)
            unique_predictions = set(y_pred)
            valid_predictions = {0, 1}
            self.assertTrue(unique_predictions.issubset(valid_predictions),
                           f"Predictions should be binary (0 or 1), got {unique_predictions}")
            logging.info(f"Model predictions are valid binary outputs: {unique_predictions}")
        except Exception as e:
            error_msg = f"Failed to validate prediction output type: {e}"
            logging.error(error_msg)
            self.fail(error_msg)

    def test_model_performance_on_holdout(self):
        """Test champion model meets minimum performance thresholds on holdout data."""
        try:
            X_holdout = self.holdout_data.iloc[:, 0:-1]
            y_holdout = self.holdout_data.iloc[:, -1]

            # predict on holdout data
            y_pred = self.model.predict(X_holdout)

            # metrics calculation
            accuracy = accuracy_score(y_holdout, y_pred)
            precision = precision_score(y_holdout, y_pred, zero_division=0)
            recall = recall_score(y_holdout, y_pred, zero_division=0)
            f1 = f1_score(y_holdout, y_pred, zero_division=0)

            # define baseline expected metrics
            expected_accuracy = 0.40
            expected_precision = 0.40
            expected_recall = 0.40
            expected_f1 = 0.40

            # assertions for performance metrics
            self.assertGreaterEqual(accuracy, expected_accuracy, 
                                   f'Accuracy {accuracy:.2f} should be at least {expected_accuracy}')
            self.assertGreaterEqual(precision, expected_precision, 
                                   f'Precision {precision:.2f} should be at least {expected_precision}')
            self.assertGreaterEqual(recall, expected_recall, 
                                   f'Recall {recall:.2f} should be at least {expected_recall}')
            self.assertGreaterEqual(f1, expected_f1, 
                                   f'F1 score {f1:.2f} should be at least {expected_f1}')
            
            logging.info(f"Champion model performance - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        except Exception as e:
            error_msg = f"Failed to evaluate model performance: {e}"
            logging.error(error_msg)
            self.fail(error_msg)

    def test_model_handles_edge_cases(self):
        """Test model handles edge cases gracefully."""
        try:
            # Test with empty-like data (after preprocessing)
            edge_cases = [
                "!!!!!!",  # only punctuation
                "123456",  # only numbers
                "   ",     # only whitespace
                "",        # empty string
            ]
            
            for test_text in edge_cases:
                processed_text = preprocess_text(test_text)
                input_data = self.vectorizer.transform([processed_text])
                input_df = pd.DataFrame(input_data.toarray())
                
                prediction = self.model.predict(input_df)
                self.assertEqual(len(prediction), 1, f"Should handle edge case: '{test_text}'")
                self.assertIn(prediction[0], [0, 1], f"Prediction should be valid for: '{test_text}'")
            
            logging.info(f"Model successfully handled {len(edge_cases)} edge cases")
        except Exception as e:
            error_msg = f"Model failed to handle edge cases: {e}"
            logging.error(error_msg)
            self.fail(error_msg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()
