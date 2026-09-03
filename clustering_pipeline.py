import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. KONFIGURASI & PATH DATASET
# ==========================================
DATASET_PATH = "data/dataset.xlsx"  # Path dataset Excel lokal
OUTPUT_DIR = "output"               # Folder output grafik & CSV hasil
N_CLUSTERS = 3                       # Jumlah klaster optimal (berdasarkan Elbow Method)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET & PREPROCESSING
# ==========================================
# Load dataset tanpa header jika struktur asli tidak memiliki nama kolom
df = pd.read_excel(DATASET_PATH, sheet_name=0, header=None)

# Beri nama kolom umum
num_cols = df.shape[1] - 2
df.columns = ["ID"] + [f"Feature_{i+1}" for i in range(num_cols)] + ["Label"]

# Pisahkan fitur dan label
df_data = df.drop(columns=["ID", "Label"]).copy()

# Cleaning data: Konversi koma ke titik dan pastikan tipe data numerik
df_data = df_data.astype(str).replace(",", ".", regex=True)
df_data = df_data.apply(pd.to_numeric, errors="coerce")
df_data.dropna(inplace=True)

# Sinkronisasi index dataframe awal
df = df.loc[df_data.index].reset_index(drop=True)
df_data = df_data.reset_index(drop=True)

# Penanganan Outlier menggunakan IQR (replacement dengan mean)
for col in df_data.columns:
    Q1 = df_data[col].quantile(0.25)
    Q3 = df_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_val = df_data[col].mean()
    df_data[col] = df_data[col].apply(
        lambda x: mean_val if x < lower_bound or x > upper_bound else x
    )

# Normalisasi dengan MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_data)

# ==========================================
# 3. DETERMINASI JUMALAH KLASTER (ELBOW METHOD)
# ==========================================
sse = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, "bo-")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/elbow_method.png")
plt.close()

# ==========================================
# 4. REDUKSI DIMENSI (PCA & LDA)
# ==========================================
# PCA (Unsupervised)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# LDA (Supervised - menggunakan Label bawaan)
df["Label_Encoded"] = pd.factorize(df["Label"])[0]
lda = LDA(n_components=2)
data_lda = lda.fit_transform(data_scaled, df["Label_Encoded"])

# ==========================================
# 5. PEMODELAN CLUSTERING (K-Means, FCM, DBSCAN)
# ==========================================
# A. K-Means
kmeans_pca = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
kmeans_labels_pca = kmeans_pca.fit_predict(data_pca)

kmeans_lda = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
kmeans_labels_lda = kmeans_lda.fit_predict(data_lda)

# B. Fuzzy C-Means (FCM)
_, u_pca, _, _, _, _, _ = fuzz.cluster.cmeans(
    data_pca.T, c=N_CLUSTERS, m=2, error=0.005, maxiter=1000
)
fcm_labels_pca = np.argmax(u_pca, axis=0)

_, u_lda, _, _, _, _, _ = fuzz.cluster.cmeans(
    data_lda.T, c=N_CLUSTERS, m=2, error=0.005, maxiter=1000
)
fcm_labels_lda = np.argmax(u_lda, axis=0)

# C. DBSCAN
dbscan_pca = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels_pca = dbscan_pca.fit_predict(data_pca)

dbscan_lda = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels_lda = dbscan_lda.fit_predict(data_lda)

# ==========================================
# 6. EVALUASI MODEL (SILHOUETTE SCORE)
# ==========================================
def calculate_silhouette(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        return -1
    return silhouette_score(X, labels)

results = {
    "Model": ["K-Means", "Fuzzy C-Means", "DBSCAN"],
    "PCA_Silhouette": [
        calculate_silhouette(data_pca, kmeans_labels_pca),
        calculate_silhouette(data_pca, fcm_labels_pca),
        calculate_silhouette(data_pca, dbscan_labels_pca),
    ],
    "LDA_Silhouette": [
        calculate_silhouette(data_lda, kmeans_labels_lda),
        calculate_silhouette(data_lda, fcm_labels_lda),
        calculate_silhouette(data_lda, dbscan_labels_lda),
    ],
}

df_eval = pd.DataFrame(results)
print("\n=== SILHOUETTE SCORE COMPARISON ===")
print(df_eval.to_string(index=False))

# Simpan hasil label klaster ke CSV
df["KMeans_PCA"] = kmeans_labels_pca
df["FCM_PCA"] = fcm_labels_pca
df["DBSCAN_PCA"] = dbscan_labels_pca
df["KMeans_LDA"] = kmeans_labels_lda
df["FCM_LDA"] = fcm_labels_lda
df["DBSCAN_LDA"] = dbscan_labels_lda

df.to_csv(f"{OUTPUT_DIR}/clustering_results.csv", index=False)
print(f"\nHasil klasterisasi dan plot berhasil disimpan di folder '{OUTPUT_DIR}/'.")