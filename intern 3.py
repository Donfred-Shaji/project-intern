import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

# Load the dataset
mcdonalds = pd.read_csv('mcdonalds.csv')

# Select columns 1 to 11
MD_x = mcdonalds.iloc[:, 0:11].values

# Convert to binary where "Yes" is 1 and anything else is 0
MD_x_binary = (MD_x == "Yes").astype(int)

# Standardize the data for K-Means clustering
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x_binary)

# Perform K-Means clustering for cluster sizes, for example, 2 to 8
n_clusters = 4  # Specify the cluster number you want to analyze (like MD.km28[["4"]])
kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
kmeans.fit(MD_x_scaled)

# Get the labels for the cluster
labels = kmeans.labels_

# Create a mask for cluster "4" (in Python, cluster labels are usually 0-indexed)
cluster_4_data = MD_x_scaled[labels == (n_clusters - 1)]  # Adjust for 0-indexing

# Plot the histogram for cluster "4"
plt.hist(cluster_4_data, bins=30, range=(0, 1), edgecolor='black', alpha=0.7)
plt.xlim(0, 1)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster 4')
plt.grid(True)
plt.show()
