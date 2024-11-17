import mlflow
import mlflow.keras
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a simple neural network model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Parameter grid to try different configurations
param_grid = [
    {"epochs": 10, "batch_size": 32},
    {"epochs": 20, "batch_size": 64},
]

# Start MLflow experiment
mlflow.set_experiment("Keras Neural Network Experiment")

for params in param_grid:
    with mlflow.start_run():
        # Create and train the model
        model = create_model()
        model.fit(X_train, y_train, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0)
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("epochs", params["epochs"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log Keras model
        model_name = f"keras_model_epochs_{params['epochs']}_batch_{params['batch_size']}"
        mlflow.keras.log_model(model, model_name)
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Logged params: {params}")
        print(f"Mean Squared Error: {mse:.4f}, R2 Score: {r2:.4f}")
