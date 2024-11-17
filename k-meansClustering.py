import mlflow
import mlflow.sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate synthetic dataset for clustering
X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=42)

# Parameter grid to explore different configurations
param_grid = [
    {"n_clusters": 3, "init": "k-means++", "random_state": 42},
    {"n_clusters": 4, "init": "k-means++", "random_state": 42},
    {"n_clusters": 5, "init": "random", "random_state": 42},
]

# Start MLflow experiment
mlflow.set_experiment("K-Means Clustering Experiment")

for params in param_grid:
    with mlflow.start_run():
        # Train K-Means Clustering model with specific parameters
        kmeans = KMeans(
            n_clusters=params["n_clusters"],
            init=params["init"],
            random_state=params["random_state"],
        )
        kmeans.fit(X)
        
        # Compute clustering metrics
        labels = kmeans.labels_
        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)

        # Log parameters and metrics
        mlflow.log_param("n_clusters", params["n_clusters"])
        mlflow.log_param("init", params["init"])
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("silhouette_score", silhouette)
        
        # Log model with unique names
        model_name = f"kmeans_clusters_{params['n_clusters']}_init_{params['init']}"
        mlflow.sklearn.log_model(kmeans, model_name)
        
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Logged params: {params}")
        print(f"Inertia: {inertia:.4f}, Silhouette Score: {silhouette:.4f}")
