# project workflow

this document contains the complete step-by-step workflow for setting up and deploying the mlops project from scratch.

## table of contents
- [1. setup project repository](#1-setup-project-repository)
- [2. setup mlflow on dagshub](#2-setup-mlflow-on-dagshub)
- [3. run ml experiments](#3-run-ml-experiments)
- [4. setup dvc project](#4-setup-dvc-project)
- [5. complete ml pipeline](#5-complete-ml-pipeline)
- [6. create dvc pipeline](#6-create-dvc-pipeline)
- [7. create flask app](#7-create-flask-app)
- [8. create dagshub token](#8-create-dagshub-token)
- [9. add tests and scripts](#9-add-tests-and-scripts)
- [10. github actions](#10-github-actions)
- [11. containerization](#11-containerization)
- [12. setup aws services](#12-setup-aws-services)
- [13. run cicd pipeline](#13-run-cicd-pipeline)
- [14. eks cluster setup](#14-eks-cluster-setup)
- [15. deploy on eks](#15-deploy-on-eks)
- [cleanup](#cleanup)

---

## 1. setup project repository

### initialize project
```bash
# create a github repo, clone it locally
git clone <your-repo-url>
cd <repo-name>

# install uv if you don't have it
pip install uv

# initialize project
uv init

# create virtual environment
uv venv .venv

# activate virtual env
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# install cookiecutter (used to create projects from templates)
pip install cookiecutter
```

### organize project structure
- rename folder and files to align with the current project structure
- git add, commit, and push changes

```bash
git add .
git commit -m "initial project setup"
git push
```

---

## 2. setup mlflow on dagshub

### create dagshub repository
- create a new dagshub repo and connect it to github
- copy mlflow tracking remote url and code snippet

**mlflow tracking remote:**
```
https://dagshub.com/aashu-0/MLOps_Learning_Project.mlflow
```

**using mlflow tracking:**
```python
import dagshub
dagshub.init(repo_owner='aashu-0', repo_name='MLOps_Learning_Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
```

### explore mlflow ui
- click on "go to mlflow ui" button and explore mlflow ui

### install dependencies
```bash
uv add dagshub mlflow
```

---

## 3. run ml experiments

- run experiments notebook in the `notebooks/` folder
- create new experiments and decide on:
  - machine learning model (in our case: logistic regression)
  - feature engineering approach (tfidf vectorizer)
  - hyperparameters to use

---

## 4. setup dvc project

### initialize dvc
```bash
dvc init
```

### add s3 as remote storage

#### setup aws resources
```bash
# login to aws console
# create an iam user with permission policies → AdministratorAccess
# create an s3 bucket (name: mshrashu-dvc-storage)
```

#### install dependencies
```bash
uv pip install dvc[s3] awscli
uv add dvc[s3] awscli
```

#### configure aws credentials
```bash
aws configure
```
provide:
- `AWS Access Key ID` → `<your-access-key>`
- `AWS Secret Access Key` → `<your-secret-key>`
- `Default region` → `<your-aws-region>`
- `Default output` → `json`

#### add s3 as dvc remote
```bash
# add s3 remote
dvc remote add -d s3remote s3://mshrashu-dvc-storage

# verify remote
dvc remote list

# remove remote (if needed)
dvc remote remove <name>

# push data to remote
dvc push
```

### alternative: local folder as remote storage
```bash
# create a local folder
mkdir local_s3

# add local remote
dvc remote add -d mylocal local_s3
```

---

## 5. complete ml pipeline

create the entire ml pipeline under the `src/` folder:

### logger setup
- create a `logger/` folder with logging configuration

### data pipeline
**`data/data_ingestion.py`**
- load data from source
- preprocess it
- split into train and test
- save to `./dataset/raw/` folder

**`data/data_preprocessing.py`**
- extra cleaning and text normalization steps on the ingested data
- save preprocessed data in `./dataset/interim/` folder

### feature engineering
**`features/feature_engineering.py`**
- apply tf-idf to text data
- save processed data in `./dataset/processed/` directory

### model pipeline
**`model/model_building.py`**
- build and save a logistic regression model using training data

**`model/model_evaluation.py`**
- evaluate trained model using test data
- log metrics to mlflow

**`model/register_model.py`**
- register trained model to the mlflow model registry

---

## 6. create dvc pipeline

### setup pipeline files
```bash
# create dvc.yaml file (pipeline definition)
# create params.yaml file (pipeline parameters)
```

### run pipeline
```bash
# reproduce dvc pipeline by running all stages
dvc repro

# commit changes
git add .
git commit -m "add dvc pipeline"
git push

# push data to dvc remote
dvc push
```

---

## 7. create flask app

### install flask
```bash
uv add flask
```

### create app structure
- create a directory `flask_app/`
- write html, css and `app.py`

### create separate requirements.txt
**why?** during containerization we only create image of app, so all other project requirements will only increase the size of our docker image

**how?**
```bash
# install pipreqs
uv pip install pipreqs

# navigate to flask_app directory
cd flask_app

# generate requirements.txt
pipreqs . --force
```

---

## 8. create dagshub token

### generate token
- go to dagshub → user settings
- under "manage personal access tokens"
- generate new token
- save token: `mlops_test` (<token_name>): `<your-dagshub-token>`

### add to github secrets
- add this token to github secrets with name `DAGSHUB_TOKEN`

---

## 9. add tests and scripts

### tests folder
**`tests/test_flask_app.py`**
- unittests for flask application

**`tests/test_model.py`**
- tests for loading and validating the ml model from mlflow registry

### scripts folder
**`scripts/promote_model.py`**
- script to promote a model from @Candidate alias to @Champion alias in the mlflow model registry

---

## 10. github actions

### create cicd workflow
- add `.github/workflows/cicd.yaml`
- configure automated testing, building, and deployment

---

## 11. containerization

### create docker image

#### start docker engine
```bash
# open docker desktop
```

#### build docker image
```bash
# in root directory, run:
docker build -t mlops-project:latest .

# run a container
docker run -p 8888:5000 -e DAGSHUB_TOKEN=<your-dagshub-token> mlops-project:latest
```

### push docker image to docker hub (optional)

#### setup
```bash
# create a repo on docker hub

# tag the image
docker tag mlops-project:latest aashu0/mlops-project:latest

# push to docker hub
docker push aashu0/mlops-project:latest
```

#### test pulled image
```bash
# delete images locally
docker rmi mlops-project:latest
docker rmi aashu0/mlops-project:latest

# pull from docker hub
docker pull aashu0/mlops-project:latest

# run a container from pulled image
docker run -p 8888:5000 -e DAGSHUB_TOKEN=<your-dagshub-token> aashu0/mlops-project:latest
```

---

## 12. setup aws services

### add credentials to github secrets
add the following secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ACCOUNT_ID`
- `AWS_REGION`
- `ECR_REPOSITORY`

### configure iam permissions
- add `AmazonEC2ContainerRegistryFullAccess` policy to permissions policies for iam user

---

## 13. run cicd pipeline

### initial deployment
```bash
# run cicd till stage "push docker image to ecr"
git add .
git commit -m "add cicd pipeline"
git push
```

---

## 14. eks cluster setup

### prerequisites checklist

verify you have installed:
- **aws cli**: command line tool to interact with aws services
- **kubectl**: command line tool for kubernetes
- **eksctl**: command line utility for amazon eks service

```bash
# check versions
aws --version
kubectl version --client
eksctl version
```

### install missing tools

#### aws cli
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html

#### kubectl
```bash
choco install kubernetes-cli -y
```

#### eksctl
```bash
choco install eksctl -y
```

#### chocolatey (if not installed)
https://docs.chocolatey.org/en-us/choco/setup/

### create eks cluster

#### configure aws
```bash
aws configure
```

#### create cluster
```bash
eksctl create cluster `
  --name mlops-cluster `
  --region eu-north-1 `
  --nodegroup-name standard-workers `
  --node-type t3.small `
  --nodes 1 `
  --nodes-min 1 `
  --nodes-max 1 `
  --managed
```

#### update kubectl config
once cluster is created, eksctl automatically updates kubectl config file

```bash
# verify kubectl config
aws eks --region eu-north-1 update-kubeconfig --name mlops-cluster

# list clusters
aws eks list-clusters
```

### verify cluster

#### check cluster status
```bash
aws eks --region eu-north-1 describe-cluster --name mlops-cluster --query "cluster.status"
```

#### check cluster connectivity
```bash
kubectl get nodes
```

#### check namespaces
```bash
kubectl get namespaces
```

#### verify deployment
```bash
kubectl get pods
kubectl get svc
```

### cluster management commands

#### delete cluster
```bash
eksctl delete cluster --name mlops-cluster --region eu-north-1
```

#### verify cluster deletion
```bash
eksctl get cluster --region eu-north-1
```

---

## 15. deploy on eks

### add deployment stages
- add next stages in `cicd.yaml`
- create `deployment.yaml` and `service.yaml`

### configure security group
- edit the security group for nodes
- add inbound rule for port 5000 to access the app

### access the application

#### get external ip
```bash
kubectl get svc mlops-project-service
```

#### access app
```bash
# browser
http://<external-ip>:5000

# or via terminal
curl http://<external-ip>:5000
```

---

## cleanup

### aws resource cleanup

#### delete kubernetes resources
```bash
# delete deployment
kubectl delete deployment mlops-project-deployment

# delete service
kubectl delete service mlops-project-service

# delete env variable
kubectl delete secret dagshub-secret
```

#### delete eks cluster
```bash
# delete cluster
eksctl delete cluster --name mlops-cluster --region eu-north-1

# verify cluster deletion
eksctl get cluster --region eu-north-1
```

#### delete aws artifacts
- delete artifacts from ecr
- delete artifacts from s3

#### validate cloudformation
- validate if cloudformation stacks are deleted

---

## notes

- always ensure docker desktop is running before building images
- keep aws credentials secure and never commit them to git
- regularly backup your dvc remote storage
- monitor aws costs, especially for eks clusters
- delete resources when not in use to avoid unnecessary charges

---

**end of workflow**