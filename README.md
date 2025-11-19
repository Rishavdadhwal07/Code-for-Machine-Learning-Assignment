# Code-for-Machine-Learning-Assignment
#This code is used in my assignment of Machine Learning Subject
Here is the code: 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

df = pd.read_csv('Fragrance Dataset - COM7022 - [4037].csv')

print("Dataset shape:", df.shape)
print("These are the descriptive statistics of the given dataset:\n",df.describe())
print("The null values in each column: \n",df.isnull().sum())

#Dropping the null values rows from brand and type columns
df = df.dropna(subset=['brand']) 
df = df.dropna(subset=['type'])

# Filling the missing values in price column 
df['price'] = df['priceWithCurrency'].str.replace(r'[^0-9.]', '', regex=True).astype(float)
df['price'] = df['price'].fillna(df['price'])

df['available'] = df['available'].fillna(0)

# Filling the missing values in sold column
df['sold'] = df['availableText'].str.extract(r'/\s*(\d+)')
df['sold'] = pd.to_numeric(df['sold'], errors='coerce').fillna(0).astype(int) 

# Drop the lastUpdated column
df = df.drop(columns=['lastUpdated'])

print("The null values in each column after data cleaning: \n",df.isnull().sum())

# ---------------- FEATURE SELECTION -----------------
selected_features = ['price', 'sold', 'available', 'type']
df_selected = df[selected_features].copy()

#Encode 'type' column (Label Encoding)
le = LabelEncoder()
df_selected['type'] = le.fit_transform(df_selected['type'])

#Handle missing values
df_selected = df_selected.dropna() 

#Scale features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)
df_scaled = pd.DataFrame(df_scaled, columns=df_selected.columns)


# -------------------------------------------------------------
# ⭐⭐ CHOOSING THE VALUE OF k (Elbow + Silhouette) ⭐⭐
# -------------------------------------------------------------
K_range = range(2, 10)

# --------- ELBOW METHOD ----------
inertia_values = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia_values, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.show()

# --------- SILHOUETTE METHOD ----------
silhouette_scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(df_scaled)
    sil = silhouette_score(df_scaled, labels)
    silhouette_scores.append(sil)
    print(f"k={k}, silhouette score={sil:.4f}")

plt.figure(figsize=(6,4))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Scores for Different k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()

# Pick best k from silhouette
best_k = K_range[np.argmax(silhouette_scores)]
print("\n🔥 Best k chosen based on silhouette score =", best_k)

# -------------------------------------------------------------

# ====== 1. KMeans Clustering ======
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(df_scaled)

df_selected['kmeans_cluster'] = kmeans_labels

print("KMeans cluster counts:\n", df_selected['kmeans_cluster'].value_counts())

# ====== 2. DBSCAN Clustering ======
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_scaled)

df_selected['dbscan_cluster'] = dbscan_labels

print("DBSCAN cluster counts:\n", df_selected['dbscan_cluster'].value_counts())

# ====== PCA Visualization ======
pca = PCA(n_components=2, random_state=42)
df_pca = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

df_pca['kmeans_cluster'] = kmeans_labels
df_pca['dbscan_cluster'] = dbscan_labels

plt.figure(figsize=(14,6))

# KMeans
plt.subplot(1,2,1)
for cluster in df_pca['kmeans_cluster'].unique():
    plt.scatter(df_pca[df_pca['kmeans_cluster']==cluster]['PC1'],
                df_pca[df_pca['kmeans_cluster']==cluster]['PC2'],
                label=f'Cluster {cluster}')
plt.title(f'KMeans Clusters (k={best_k})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

# DBSCAN
plt.subplot(1,2,2)
for cluster in df_pca['dbscan_cluster'].unique():
    plt.scatter(df_pca[df_pca['dbscan_cluster']==cluster]['PC1'],
                df_pca[df_pca['dbscan_cluster']==cluster]['PC2'],
                label=f'Cluster {cluster}')
plt.title('DBSCAN Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.tight_layout()
plt.show()

# ====== Evaluation ======
print("KMeans Silhouette Score:", silhouette_score(df_scaled, kmeans_labels))
print("KMeans Davies-Bouldin Index:", davies_bouldin_score(df_scaled, kmeans_labels))

db_points = df_scaled[dbscan_labels != -1]
db_labels = dbscan_labels[dbscan_labels != -1]

if len(db_points) > 1:
    print("DBSCAN Silhouette Score:", silhouette_score(db_points, db_labels))
    print("DBSCAN Davies-Bouldin Index:", davies_bouldin_score(db_points, db_labels))
