import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid to try different configurations
param_grid = [
    {"criterion": "squared_error", "max_depth": 5, "min_samples_split": 2},
    {"criterion": "squared_error", "max_depth": 10, "min_samples_split": 4},
    {"criterion": "absolute_error", "max_depth": None, "min_samples_split": 2},
]

# Start MLflow experiment
mlflow.set_experiment("Decision Tree Regressor Experiment")

for params in param_grid:
    with mlflow.start_run():
        # Train Decision Tree Regressor model with specific parameters
        reg = DecisionTreeRegressor(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        )
        reg.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("criterion", params["criterion"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("min_samples_split", params["min_samples_split"])
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model with unique names
        model_name = f"decision_tree_reg_{params['criterion']}_depth_{params['max_depth']}_split_{params['min_samples_split']}"
        mlflow.sklearn.log_model(reg, model_name)
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Logged params: {params}")
        print(f"Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}")
