import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Load the dataset (assuming it's in CSV or a similar format)
mcdonalds = pd.read_csv('mcdonalds.csv')
np.random.seed(1234)

# Get the column names
mcdonalds.columns
mcdonalds.shape
print(mcdonalds.shape)
print(mcdonalds.head(3))
MD_x = mcdonalds.iloc[:, 0:11].values

# Convert to binary where "Yes" is 1 and anything else is 0
MD_x_binary = (MD_x == "Yes").astype(int)

# Calculate column means and round to 2 decimal places
column_means = np.round(np.mean(MD_x_binary, axis=0), 2)

print(column_means)
pca = PCA()
MD_pca = pca.fit(MD_x_binary)

# Summarize PCA
explained_variance_ratio = np.round(pca.explained_variance_ratio_, 4)
cumulative_variance = np.round(np.cumsum(pca.explained_variance_ratio_), 4)

# Display summary
print("Explained Variance by Each Component:", explained_variance_ratio)
print("Cumulative Explained Variance:", cumulative_variance)
components = np.round(pca.components_, 1)
explained_variance = np.round(pca.explained_variance_, 1)
explained_variance_ratio = np.round(pca.explained_variance_ratio_, 1)

# Print PCA details with digits rounded to 1
print("PCA Components (rounded):")
print(components)

print("\nExplained Variance (rounded):")
print(explained_variance)

print("\nExplained Variance Ratio (rounded):")
print(explained_variance_ratio)
MD_pca_transformed = pca.transform(MD_x_binary)

# Plot the PCA results in grey
plt.scatter(MD_pca_transformed[:, 0], MD_pca_transformed[:, 1], color='grey')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.grid(True)

# Project the axes (approximation of projAxes in R)
# Plot the loadings (principal component vectors)
for i, (pc1, pc2) in enumerate(zip(pca.components_[0], pca.components_[1])):
    plt.arrow(0, 0, pc1, pc2, color='r', alpha=0.5, head_width=0.05)
    plt.text(pc1, pc2, f"Var{i+1}", color='r')

plt.show()
