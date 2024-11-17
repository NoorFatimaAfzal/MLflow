import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid to try different configurations
param_grid = [
    {"criterion": "gini", "max_depth": 3, "min_samples_split": 2},
    {"criterion": "entropy", "max_depth": 5, "min_samples_split": 4},
    {"criterion": "gini", "max_depth": None, "min_samples_split": 2},
]

# Start MLflow experiment
mlflow.set_experiment("Decision Tree Classifier Experiment")

for params in param_grid:
    with mlflow.start_run():
        # Train Decision Tree model with specific parameters
        clf = DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        )
        clf.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        
        # Log parameters and metrics
        mlflow.log_param("criterion", params["criterion"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("min_samples_split", params["min_samples_split"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model with unique names based on parameters
        model_name = f"decision_tree_{params['criterion']}_depth_{params['max_depth']}_split_{params['min_samples_split']}"
        mlflow.sklearn.log_model(clf, model_name)

        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Logged params: {params}")
        print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
