import mlflow
import mlflow.sklearn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameter grid to explore different configurations
param_grid = [
    {"n_neighbors": 3, "weights": "uniform"},
    {"n_neighbors": 5, "weights": "uniform"},
    {"n_neighbors": 3, "weights": "distance"},
    {"n_neighbors": 5, "weights": "distance"},
]

# Start MLflow experiment
mlflow.set_experiment("KNN Regression Experiment")

for params in param_grid:
    with mlflow.start_run():
        # Train KNN Regressor with specific parameters
        knn = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"], 
            weights=params["weights"]
        )
        knn.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = knn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("n_neighbors", params["n_neighbors"])
        mlflow.log_param("weights", params["weights"])
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log model with unique names
        model_name = f"knn_reg_{params['weights']}_k_{params['n_neighbors']}"
        mlflow.sklearn.log_model(knn, model_name)
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Logged params: {params}")
        print(f"Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}")
