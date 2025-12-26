import string
import pandas as pd
import numpy as np
import re
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# setup MLflow & DAGSHUB
MLFLOW_TRACKING_URI = "https://dagshub.com/aashu-0/MLOps_Learning_Project.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
dagshub.init(repo_owner = "aashu-0", repo_name = "MLOps_Learning_Project", mlflow = True)
mlflow.set_experiment("Logistic Regression Hyperparameter Tuning")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# text preprocessing function
def text_preprocessing(text):
    """Complete text preprocessing pipeline."""
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

# load and prepare data
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    df["review"] = df["review"].astype(str).apply(text_preprocessing)
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
    
    #vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer

# train and log model
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, scoring='f1', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # log hyperparameters and metrics
        for param, mean_score, std_score in zip(grid_search.cv_results_['params'],
                                                 grid_search.cv_results_['mean_test_score'],
                                                 grid_search.cv_results_['std_test_score']):
            with mlflow.start_run(run_name = f"Logistic Regression with params: {param}", nested=True):
                model = LogisticRegression(**param)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                #log params and metrics
                mlflow.log_params(param)
                mlflow.log_metrics(metrics)

                print(f"Logged model with params: {param} and Accuracy: {metrics['accuracy']}")

        # log the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "best_logistic_regression_model")
        print(f"Best Model Params: {best_params} with F1 Score: {best_f1:.4f}")

# main
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test, vectorizer) = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)


# Best Model Params: {'C': 10, 'max_iter': 200, 'penalty': 'l2', 'solver': 'saga'} with F1 Score: 0.7679