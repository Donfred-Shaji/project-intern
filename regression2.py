import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset
mcdonalds = pd.read_csv('mcdonalds.csv')

# Convert the 'Like' column to numeric and create 'Like.n'
mcdonalds['Like_numeric'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')
mcdonalds['Like.n'] = 6 - mcdonalds['Like_numeric']

# Select relevant features (first 11 columns)
features = mcdonalds.iloc[:, 0:11]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Fit Gaussian Mixture Model with 2 components
gmm = GaussianMixture(n_components=2, random_state=1234, n_init=10)
gmm.fit(features_scaled)

# Get cluster labels
cluster_labels = gmm.predict(features_scaled)

# Count cluster sizes
cluster_sizes = np.bincount(cluster_labels)

# Display results similar to R output
print("Cluster sizes:")
for idx, size in enumerate(cluster_sizes, start=1):
    print(f"{idx}: {size}")

# Convergence information
print(f"\nConvergence after {gmm.converged_} iterations")
