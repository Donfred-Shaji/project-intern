import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(1234)

# Load the dataset
mcdonalds = pd.read_csv('mcdonalds.csv')

# Select columns 1 to 11
MD_x = mcdonalds.iloc[:, 0:11].values

# Convert to binary where "Yes" is 1 and anything else is 0
MD_x_binary = (MD_x == "Yes").astype(int)

# Standardize the data for better convergence
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)

# Initialize lists to hold AIC and BIC values
aic_values = []
bic_values = []
cluster_sizes = range(2, 9)  # k = 2 to 8

# Fit Gaussian Mixture Models for k=2 to k=8
for n_clusters in cluster_sizes:
    gmm = GaussianMixture(n_components=n_clusters, random_state=1234)
    gmm.fit(MD_x_scaled)
    
    # Store AIC and BIC
    aic_values.append(gmm.aic(MD_x_scaled))
    bic_values.append(gmm.bic(MD_x_scaled))

# Plot AIC and BIC
plt.figure(figsize=(10, 6))
plt.plot(cluster_sizes, aic_values, marker='o', linestyle='-', color='blue', label='AIC')
plt.plot(cluster_sizes, bic_values, marker='o', linestyle='--', color='red', label='BIC')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Value of Information Criteria')
plt.title('AIC and BIC for Different Number of Clusters')
plt.xticks(cluster_sizes)  # Set x-ticks to cluster sizes
plt.grid(True)
plt.legend()
plt.show()
