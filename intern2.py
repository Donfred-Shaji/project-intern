import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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

# Store the inertia for different cluster numbers
inertia_values = []

# Perform K-Means clustering for cluster numbers 2 to 8, repeated 10 times for consistency
for n_clusters in range(2, 9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
    kmeans.fit(MD_x_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the number of clusters (segments) vs inertia using a bar chart
plt.bar(range(2, 9), inertia_values, color='b', alpha=0.7)

# Label the chart
plt.xlabel('Number of Segments')
plt.ylabel('sum of Within-distance)')
plt.title('K-Means Clustering: Number of Segments vs Inertia')
plt.grid(axis='y')

# Show the bar plot
plt.show()
