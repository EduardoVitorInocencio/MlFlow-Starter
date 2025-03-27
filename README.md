# MLflow Logistic Regression Model Tracking and Deployment

This repository demonstrates the usage of **MLflow** for tracking machine learning experiments, logging models, and validating them with serving payloads. The example uses the Iris dataset and Logistic Regression from **scikit-learn** to predict flower species based on various features.

## Introduction to MLflow

MLflow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides tools for tracking experiments, packaging code into reproducible runs, and sharing models across different teams. With MLflow, data scientists and machine learning engineers can efficiently manage their workflows from experimentation to deployment.

### When to Use MLflow

MLflow is particularly beneficial in scenarios where machine learning models need to be tracked, managed, and versioned in a collaborative environment. It is ideal when:

- **Experimentation and Tracking**: You need to track different model training experiments, record parameters, metrics, and results to compare performance.
- **Reproducibility**: Ensuring that your machine learning workflows are reproducible, meaning that the same experiment can be reproduced by another team member or at a later time.
- **Model Deployment**: You are looking to deploy models to production, ensuring they are packaged properly and can be served via an API.
- **Model Management**: You need to manage and organize multiple models, track their versions, and integrate them into production systems.
- **Collaboration**: If your team works collaboratively on machine learning projects and you need a shared platform for version control and tracking.

### Use Cases for MLflow

1. **Experiment Tracking**: MLflow allows you to log the parameters, metrics, and output of each model training session. It provides a convenient UI for comparing different experiments to choose the best-performing model.

2. **Model Versioning**: MLflow’s model registry allows you to version your models, track their lifecycle, and organize models into stages like "staging", "production", and "archived". This helps in managing multiple versions and deploying them in an organized manner.

3. **Reproducible Pipelines**: MLflow can record the entire machine learning pipeline, including the preprocessing steps, hyperparameters, and the model itself, ensuring that all experiments are reproducible.

4. **Model Packaging and Deployment**: MLflow allows you to package models in standard formats (e.g., Python functions, TensorFlow, PyTorch) and deploy them as REST APIs or integrate them with different serving platforms like AWS SageMaker or Azure ML.

5. **Collaboration and Model Sharing**: With MLflow, you can easily share models with teammates by registering them in the model registry. It helps ensure that models are consistently deployed, managed, and accessed by the team.

By using MLflow, machine learning teams can focus more on model development and less on tracking, versioning, and deployment logistics, improving efficiency and collaboration in the ML lifecycle.


## Theory of Each Command

- **`import mlflow`**: Imports the MLflow library for managing machine learning workflows.
- **`import pandas as pd`**: Imports the Pandas library for data manipulation and analysis.
- **`from mlflow.models import infer_signature`**: Imports `infer_signature` to generate the model signature, which describes the input and output schema.
- **`from sklearn import datasets`**: Imports the `datasets` module from scikit-learn to load pre-existing datasets.
- **`from sklearn.linear_model import LogisticRegression`**: Imports the logistic regression model from scikit-learn.
- **`from sklearn.metrics import accuracy_score`**: Imports the accuracy score metric to evaluate the model's performance.
- **`from sklearn.model_selection import train_test_split`**: Imports the `train_test_split` function to split the dataset into training and testing sets.
- **`mlflow.set_tracking_uri("http://127.0.0.1:5000")`**: Sets the URI of the MLflow tracking server where the experiment will be logged.
- **`X, y = datasets.load_iris(return_X_y=True)`**: Loads the Iris dataset, splitting it into feature variables (`X`) and target labels (`y`).
- **`train_test_split(X, y, test_size=0.25, random_state=42)`**: Splits the dataset into training and test sets, with 25% of the data used for testing.
- **`params = {...}`**: Defines the hyperparameters for the logistic regression model.
- **`lr = LogisticRegression(**params)`**: Initializes the logistic regression model with the specified hyperparameters.
- **`lr.fit(X_train, y_train)`**: Trains the logistic regression model on the training dataset.
- **`y_pred = lr.predict(X_test)`**: Makes predictions on the test set using the trained model.
- **`accuracy = accuracy_score(y_test, y_pred)`**: Calculates the accuracy of the model on the test set.
- **`mlflow.set_experiment("MLFLOW QuickStart")`**: Creates or sets an experiment in MLflow where the results will be stored.
- **`with mlflow.start_run():`**: Starts a new MLflow run to track and log metrics, parameters, and models.
- **`mlflow.log_params(params)`**: Logs the hyperparameters used to train the model.
- **`mlflow.log_metric("accuracy", accuracy)`**: Logs the accuracy metric from the model.
- **`mlflow.set_tag("Training Info", "Basic LR model for iris data")`**: Adds a tag to the experiment for future identification and reference.
- **`signature = infer_signature(X_train, lr.predict(X_train))`**: Infers the input-output signature of the model based on the training data and predictions.
- **`model_info = mlflow.sklearn.log_model(...)`**: Logs the trained model into MLflow with details like signature, example input, and registered model name.
- **`validate_serving_input(model_uri, serving_payload)`**: Validates the model's serving input format with the provided JSON payload.
- **`loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)`**: Loads the logged model from MLflow using its model URI.
- **`predictions = loaded_model.predict(X_test)`**: Makes predictions using the loaded model.
- **`result = pd.DataFrame(X_test, columns=iris_features_name)`**: Creates a DataFrame to hold the test features with their respective names.
- **`model_info = mlflow.sklearn.log_model(...)`**: Logs the model again for tracking.
- **`mlflow.end_run()`**: Ends the current MLflow run.
- **`mlflow.set_experiment("MLFLOW Quickstart")`**: Creates or sets a new experiment again.
- **`mlflow.start_run()`**: Starts a new run for tracking in the current experiment.
- **`model_uri = f"models:/{model_name}/{model_version}"`**: Builds the URI to retrieve a specific version of the model from the MLflow model registry.
- **`model = mlflow.sklearn.load_model(model_uri)`**: Loads the model from the model registry using its URI.

## Code Breakdown

### 1. **Import Libraries**
```python
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
```
- **mlflow**: For managing machine learning experiments.
- **pandas**: For data manipulation.
- **scikit-learn**: To load the Iris dataset, train a logistic regression model, and compute accuracy.

### 2. **Set Tracking URI**
```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```
- Sets the URI for the MLflow tracking server. This points to a local server running at `http://127.0.0.1:5000`.

### 3. **Load the Dataset**
```python
X, y = datasets.load_iris(return_X_y=True)
```
- Loads the Iris dataset, with `X` as features and `y` as target labels.

### 4. **Split Data Into Training and Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
- Splits the dataset into training and testing sets with 25% of the data reserved for testing.

### 5. **Define Hyperparameters**
```python
params = {"penalty":"l2", "solver":"lbfgs", "max_iter":1000, "multi_class":"auto", "random_state":8888}
```
- Specifies the hyperparameters for the logistic regression model, such as the penalty type, solver, and the number of iterations.

### 6. **Train the Logistic Regression Model**
```python
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
```
- Creates and trains the logistic regression model on the training data.

### 7. **Make Predictions and Compute Accuracy**
```python
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```
- Makes predictions on the test set and computes the accuracy of the model.

### 8. **Set up MLflow Experiment**
```python
mlflow.set_experiment("MLFLOW QuickStart")
```
- Creates a new MLflow experiment called "MLFLOW QuickStart".

### 9. **Start MLflow Run**
```python
with mlflow.start_run():
```
- Begins tracking a new run in the MLflow experiment.

### 10. **Log Hyperparameters**
```python
mlflow.log_params(params)
```
- Logs the hyperparameters used for training the logistic regression model.

### 11. **Log Accuracy Metric**
```python
mlflow.log_metric("accuracy", accuracy)
```
- Logs the accuracy of the model as a metric in MLflow.

### 12. **Set Run Tags**
```python
mlflow.set_tag("Training Info", "Basic LR model for iris data")
```
- Adds a tag to the run for organizational purposes.

### 13. **Infer Model Signature**
```python
signature = infer_signature(X_train, lr.predict(X_train))
```
- Infers the model's signature, which describes the input and output formats for the model.

### 14. **Log the Model**
```python
model_info = mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="iris_model",
    signature=signature,
    input_example=X_train,
    registered_model_name="tracking_quickstart"
)
```
- Logs the logistic regression model into MLflow with the defined signature and input example for future deployment.

### 15. **Validate Serving Payload**
```python
serving_payload = """{ ... }"""  # JSON Payload
validate_serving_input(model_uri, serving_payload)
```
- Defines a JSON payload and validates it against the model for serving predictions.

### 16. **Load the Model for Inference**
```python
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)
```
- Loads the model from the MLflow server and uses it to make predictions on the test set.

### 17. **Display Predictions**
```python
iris_features_name = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_features_name)
result["actual_class"] = y_test
result['predicted_class'] = predictions
result[:5]
```
- Creates a DataFrame to display the actual and predicted classes for the first few rows of the test set.

### 18. **Log Model Again**
```python
model_info = mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="iris_model",
    signature=signature,
    input_example=X_train,
    registered_model_name="tracking-quickstart"
)
```
- Logs the model again with a different registered model name.

### 19. **Load the Latest Model Version**
```python
model_name = "tracking_quickstart"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
```
- Loads the latest version of the registered model from MLflow for further inference.

### Conclusion
This script demonstrates how to use MLflow for end-to-end machine learning model tracking, including logging hyperparameters, metrics, and models, as well as validating inputs for serving and loading the model for inference. It provides an easy way to track and manage machine learning models, making it easier to monitor and deploy models in production environments.