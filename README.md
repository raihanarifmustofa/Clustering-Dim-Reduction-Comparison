# Comparative Study of Clustering Algorithms with Dimensionality Reduction (PCA vs LDA)

Proyek ini membandingkan performa tiga algoritma pengelompokan (*clustering*): **K-Means**, **Fuzzy C-Means (FCM)**, dan **DBSCAN** yang dikombinasikan dengan dua teknik reduksi dimensi: **Principal Component Analysis (PCA)** dan **Linear Discriminant Analysis (LDA)**.

## Metodologi Pipeline

1. **Preprocessing & Cleaning**:
   - Sanitasi format angka numerik dan pembersihan *missing values*.
   - Imputasi nilai pencilan (*outliers*) berbasis rentang IQR menggunakan mean.
   - Normalisasi rentang fitur menggunakan `MinMaxScaler`.
2. **Reduksi Dimensi**:
   - **PCA**: Ekstraksi fitur secara *unsupervised*.
   - **LDA**: Ekstraksi fitur secara *supervised* berbasis label awal.
3. **Pemodelan Klasterisasi**:
   - K-Means (ditentukan via *Elbow Method*).
   - Fuzzy C-Means (soft clustering).
   - DBSCAN (density-based clustering).
4. **Evaluasi**:
   - Perbandingan metrik **Silhouette Score** untuk tiap kombinasi reduksi dimensi dan algoritma klaster.

## Instalasi & Penggunaan

1. **Clone Repositori**:
   ```bash
   git clone [https://github.com/USERNAME_KAMU/clustering-dim-reduction-comparison.git](https://github.com/USERNAME_KAMU/clustering-dim-reduction-comparison.git)
   cd clustering-dim-reduction-comparison
