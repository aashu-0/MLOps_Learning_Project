# IMDb Sentiment Analysis - MLOps Project
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-blue?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-lightblue?logo=mlflow)
![DagsHub](https://img.shields.io/badge/DagsHub-ML%20Collaboration-purple)

![Docker](https://img.shields.io/badge/Docker-Containerization-blue?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-blue?logo=kubernetes&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud-orange?logo=amazonaws&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black?logo=githubactions)
![Flask](https://img.shields.io/badge/Flask-API-black?logo=flask)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange?logo=prometheus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


An end-to-end MLOps project demonstrating sentiment analysis on IMDb movie reviews with complete CI/CD pipeline, experiment tracking, model registry, and production deployment on AWS EKS.

## Project Overview

This project implements a production-grade machine learning pipeline for binary sentiment classification (positive/negative) of movie reviews. It showcases modern MLOps practices including:

- **Automated ML Pipeline** with DVC for reproducibility
- **Experiment Tracking** with MLflow and DagsHub
- **Model Registry** with versioning and alias management
- **CI/CD Pipeline** with GitHub Actions
- **Containerized Deployment** on AWS EKS with Kubernetes
- **Monitoring** with Prometheus metrics
- **Automated Testing** for model validation

## Architecture

```
┌─────────────────┐
│   Data Source   │
│  (GitHub/S3)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DVC Pipeline   │
│  ├─ Ingestion   │
│  ├─ Preprocess  │
│  ├─ Features    │
│  ├─ Training    │
│  └─ Evaluation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ MLflow Tracking │◄────►│   DagsHub    │
│ Model Registry  │      │  (Remote)    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  GitHub Actions │
│   CI/CD Pipeline│
│  ├─ Tests       │
│  ├─ Build       │
│  └─ Deploy      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Docker/ECR    │─────►│   AWS EKS    │
│  Container Reg. │      │  Kubernetes  │
└─────────────────┘      └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │ Flask Web App│
                         │ + Prometheus │
                         └──────────────┘
```

## Features

### ML Pipeline
- **Data Ingestion**: Automated data loading and train-test split
- **Preprocessing**: Text cleaning, stopword removal, and lemmatization
- **Feature Engineering**: TF-IDF vectorization with configurable parameters
- **Model Training**: Logistic Regression with optimized hyperparameters
- **Evaluation**: Comprehensive metrics (accuracy, precision, recall, AUC)

### MLOps Components
- **Version Control**: Git + DVC for data and model versioning
- **Experiment Tracking**: MLflow integration with DagsHub
- **Model Registry**: Automated model registration with alias management (@Candidate/@Champion)
- **CI/CD**: Automated testing, building, and deployment
- **Monitoring**: Prometheus metrics for predictions and API performance
- **Testing**: Unit tests for model performance and API endpoints

## Prerequisites

- Python 3.12+
- AWS Account (for S3 and EKS deployment)
- DagsHub Account (for MLflow tracking)
- Docker
- kubectl (for Kubernetes management)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aashu-0/MLOps_Learning_Project.git
cd MLOps_Learning_Project
```

### 2. Set Up Python Environment

Using `uv` (recommended):
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# DagsHub/MLflow
DAGSHUB_TOKEN=your_dagshub_token_here

# AWS (for deployment)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your_account_id
ECR_REPOSITORY=mlops-project
EKS_CLUSTER_NAME=mlops-cluster
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🎮 Usage

### Running the DVC Pipeline

Execute the complete ML pipeline:

```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro feature_engineering

# Visualize pipeline
dvc dag
```

### Training the Model Locally

```bash
# Data ingestion
python src/data/data_ingestion.py

# Preprocessing
python src/data/data_preprocessing.py

# Feature engineering
python src/features/feature_engineering.py

# Model training
python src/model/model_building.py

# Model evaluation
python src/model/model_evaluation.py

# Register model
python src/model/register_model.py
```

### Running the Flask Application

```bash
# Local development
cd flask_app
python app.py

# With Docker
docker build -t mlops-project .
docker run -p 5000:5000 -e DAGSHUB_TOKEN=$DAGSHUB_TOKEN mlops-project
```

Access the application at `http://localhost:5000`

### Testing

```bash
# Run all tests
uv run python -m unittest discover tests

# Run specific test suite
uv run python -m unittest tests/test_model.py
uv run python -m unittest tests/test_flask_app.py
```

## Project Structure

```
MLOps_Learning_Project/
├── .dvc/                                  # DVC configuration
├── .github/workflows/                     # CI/CD pipelines
│   └── cicd.yaml                          # Main CI/CD workflow
├── dataset/                               # Data directories (DVC tracked)
│   ├── raw/                               # Raw data
│   ├── interim/                           # Preprocessed data
│   └── processed/                         # Feature-engineered data
├── flask_app/                             # Web application
│   ├── app.py                             # Flask application
│   ├── templates/                         # HTML templates
│   └── requirements.txt                   # App dependencies
├── models/                                # Trained models
│   ├── model.pkl                          # Trained model
│   └── vectorizer.pkl                     # TF-IDF vectorizer
├── notebooks/
│   ├── 1expLogisticBaseline.ipynb         # Logistic Regression baseline experiment
│   ├── 2expVectorizerAlgos.py             # Vectorization algorithms comparison
│   ├── 3expLogRegTfidf.py                 # Logistic Regression with TF-IDF tuning
│   ├── data.csv                           # Sample/raw dataset for experiments
│   └── IMDB.csv                           # IMDb movie reviews dataset
├── reports/                               # Experiment reports
│   ├── metrics.json                       # Model metrics
│   └── experiment_info.json               # MLflow run info
├── scripts/                               # Utility scripts
│   └── promote_model.py                   # Model promotion script
├── src/                                   # Source code
│   ├── data/                              # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_ingestion.py              # load and split data
│   │   └── data_preprocessing.py          # text cleaning
│   ├── features/                          # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py         # tf-idf vectorization
│   ├── model/                             # Model training & evaluation
│   │   ├── __init__.py
│   │   ├── model_building.py              # train model
│   │   ├── model_evaluation.py            # evaluate and log to mlflow
│   │   └── register_model.py              # register to mlflow registry
│   ├── logger/                            
│   │   └── __init__.py                    # logging configuration
│   └── visualization/                     # Visualization utilities
├── tests/                                 # Test suite
│   ├── test_model.py                      # Model tests
│   └── test_flask_app.py                  # API tests
├── deployment.yaml                        # Kubernetes deployment config
├── service.yaml                           # Kubernetes service config
├── dockerfile                             # Docker configuration
├── dvc.yaml                               # DVC pipeline definition
├── params.yaml                            # Pipeline parameters
└── pyproject.toml                         # Project dependencies
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/cicd.yaml`) automates:

1. **Testing**
   - Run DVC pipeline
   - Execute model validation tests
   - Run Flask application tests

2. **Model Promotion**
   - Promote validated model from @Candidate to @Champion

3. **Build & Push**
   - Build Docker image
   - Push to Amazon ECR

4. **Deploy**
   - Update Kubernetes deployment on AWS EKS
   - Apply configuration with kubectl

## Configuration

### DVC Parameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.20

feature_engineering:
  max_features: 50
```

### Model Hyperparameters

Defined in `src/model/model_building.py`:
- Algorithm: Logistic Regression
- Regularization: C=10
- Solver: saga
- Max iterations: 200

## Monitoring

The Flask application exposes Prometheus metrics at `/metrics`:

- `app_request_count`: Total requests by method and endpoint
- `app_request_latency_seconds`: Request latency histogram
- `model_prediction_count`: Prediction counts by class

**Reproducibility Guide**: A detailed end-to-end workflow is documented in **[workflow_doc.md](./workflow_doc.md)**.

## Author

**Aashutosh Mishra**
- GitHub: [@aashu-0](https://github.com/aashu-0)
- DagsHub: [aashu-0](https://dagshub.com/aashu-0)