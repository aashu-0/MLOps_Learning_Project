from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import logging
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub

# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

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

# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/aashu-0/MLOps_Learning_Project.mlflow')
dagshub.init(repo_owner='aashu-0', repo_name='MLOps_Learning_Project', mlflow=True)
# -------------------------------------------------------------------------------------

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
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Initialize Flask app
app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# Model and vectorizer setup
model_name = "LR_Classifier"

def get_latest_model_version(model_name):
    """Get the latest model version using aliases (non-deprecated approach)."""
    client = mlflow.MlflowClient()
    try:
        # Try to get model with 'staging' alias (modern approach)
        registered_model = client.get_registered_model(model_name)
        # Check for staging alias first, then production
        for alias_name in ["staging", "production"]:
            if alias_name in registered_model.aliases:
                return registered_model.aliases[alias_name]
        # If no aliases, return the latest version
        if registered_model.latest_versions:
            return registered_model.latest_versions[0].version
    except Exception as e:
        logging.warning(f"Could not retrieve registered model {model_name}: {e}")
    return None

model_version = get_latest_model_version(model_name)
if model_version:
    model_uri = f'models:/{model_name}/{model_version}'
    print(f"Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
else:
    logging.error(f"Could not find model version for {model_name}")
    raise ValueError(f"Model {model_name} not found in registry")
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    # Clean text
    text = preprocess_text(text)
    # Convert to features
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray())

    # Predict
    result = model.predict(features_df)
    prediction = result[0]

    # Increment prediction count metric
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    # Measure latency
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker
