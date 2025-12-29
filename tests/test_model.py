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
    """Test suite for the @champion aliased model"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the champion model."""
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

        # Load the model from MLflow model registry using @champion alias
        cls.model_name = "LR_Classifier"
        cls.alias = "champion"
        cls.model_uri = f'models:/{cls.model_name}@{cls.alias}'
        
        try:
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
            logging.info(f"Successfully loaded model {cls.model_name} with @{cls.alias} alias")
        except Exception as e:
            raise RuntimeError(f"Failed to load champion model with @{cls.alias} alias: {e}")

        # Load the vectorizer
        try:
            cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError("Vectorizer not found at models/vectorizer.pkl")

        # Load holdout test data
        try:
            cls.holdout_data = pd.read_csv('dataset/processed/test_tfidf.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Holdout test data not found at dataset/processed/test_tfidf.csv")

    def test_champion_alias_exists(self):
        """Test that the @champion alias exists for the model."""
        client = mlflow.MlflowClient()
        try:
            registered_model = client.get_registered_model(self.model_name)
            aliases = [alias.alias for alias in registered_model.aliases]
            self.assertIn(self.alias, aliases, f"@{self.alias} alias not found for model {self.model_name}")
            logging.info(f"@{self.alias} alias exists for {self.model_name}")
        except Exception as e:
            self.fail(f"Failed to verify @{self.alias} alias: {e}")

    def test_model_can_be_loaded_with_alias(self):
        """Test that the champion model can be loaded successfully via alias."""
        try:
            model = mlflow.pyfunc.load_model(self.model_uri)
            self.assertIsNotNone(model, "Model should load successfully with @champion alias")
            logging.info(f"Successfully loaded {self.model_name} with @{self.alias} alias")
        except Exception as e:
            self.fail(f"Failed to load model with @{self.alias} alias: {e}")

    def test_model_input_signature(self):
        """Test model accepts vectorized input and produces valid predictions."""
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

    def test_model_prediction_output_type(self):
        """Test that model predictions are valid (0 or 1 for binary classification)."""
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_pred = self.model.predict(X_holdout)
        
        # Check predictions are binary (0 or 1)
        unique_predictions = set(y_pred)
        valid_predictions = {0, 1}
        self.assertTrue(unique_predictions.issubset(valid_predictions),
                       f"Predictions should be binary (0 or 1), got {unique_predictions}")

    def test_model_performance_on_holdout(self):
        """Test champion model meets minimum performance thresholds on holdout data."""
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
        
        logging.info(f"✓ Champion model performance - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

    def test_model_handles_edge_cases(self):
        """Test model handles edge cases gracefully."""
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
            
            try:
                prediction = self.model.predict(input_df)
                self.assertEqual(len(prediction), 1, f"Should handle edge case: '{test_text}'")
                self.assertIn(prediction[0], [0, 1], f"Prediction should be valid for: '{test_text}'")
            except Exception as e:
                self.fail(f"Model failed to handle edge case '{test_text}': {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()
