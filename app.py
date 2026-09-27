# Customer Segmentation using K-Means Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# --------------------------------------------------
# 2. Select Features
# --------------------------------------------------

# We use Annual Income and Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features:")
print(X.head())

# --------------------------------------------------
# 3. Check for Missing Values
# --------------------------------------------------

print("\nMissing Values:")
print(X.isnull().sum())

# --------------------------------------------------
# 4. Standardize the Data
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 5. Find Optimal Number of Clusters
#    Using Elbow Method
# --------------------------------------------------

inertia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.grid(True)
plt.show()

# --------------------------------------------------
# 6. Apply K-Means
# --------------------------------------------------

# For this dataset, we use 5 clusters
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataset
df['Cluster'] = clusters

# --------------------------------------------------
# 7. Display Results
# --------------------------------------------------

print("\nCustomer Clusters:")
print(df.head(20))

# Number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())

# --------------------------------------------------
# 8. Visualize Customer Clusters
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=100
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")

plt.show()

# --------------------------------------------------
# 9. Calculate Cluster Centers
# --------------------------------------------------

centers_scaled = kmeans.cluster_centers_

# Convert centers back to original scale
centers = scaler.inverse_transform(centers_scaled)

print("\nCluster Centers:")

for i, center in enumerate(centers):
    print(
        f"Cluster {i}: "
        f"Income = {center[0]:.2f} k$, "
        f"Spending Score = {center[1]:.2f}"
    )

# --------------------------------------------------
# 10. Save Results
# --------------------------------------------------

df.to_csv("customer_clusters.csv", index=False)

print("\nClustered dataset saved as customer_clusters.csv")