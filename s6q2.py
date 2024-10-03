# Q2. Write a python program to implement hierarchical Agglomerative clusteringalgorithm.
# (Download Customer.csv dataset from github.com).


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the customer data from the CSV file
data = pd.read_csv("customer.csv")

# Select the features for clustering (e.g., "Age" and "Income")
X = data[["Age", "Income"]]

# Perform hierarchical agglomerative clustering
n_clusters = 2  # Number of clusters to create
linkage_type = 'ward'  # Linkage method

model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
labels = model.fit_predict(X)

# Plot the dendrogram (optional)
linkage_matrix = linkage(X, method=linkage_type)
dendrogram(linkage_matrix, orientation="top")
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Add cluster labels to the original dataset
data["Cluster"] = labels

# Print the resulting clusters
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    cluster_data = data[data["Cluster"] == cluster]
    print(cluster_data)

# Visualize the clusters
for cluster in range(n_clusters):
    cluster_data = data[data["Cluster"] == cluster]
    plt.scatter(cluster_data["Age"], cluster_data["Income"], label=f"Cluster {cluster}")

plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.title("Hierarchical Agglomerative Clustering")
plt.show()
