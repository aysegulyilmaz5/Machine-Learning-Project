import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# Veri Kümesi Yükleme
data = pd.read_csv('3.csv')

# Eksik değerleri kaldırma
data = data.dropna()

# Sayısal özellikleri seçme
numeric_data = data.select_dtypes(include=[np.number])

# Kümeleme (Clustering) - K-means
kmeans = KMeans(n_clusters=4, n_init=10)  # Siz bir mantıklı sayı belirleyebilirsiniz
kmeans.fit(numeric_data)
kmeans_labels = kmeans.labels_

# Kümeleme (Clustering) - DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(numeric_data)

# Negatif etiketleri pozitif değerlere dönüştürme
dbscan_labels[dbscan_labels == -1] = 0

# Sınıflandırma (Classification) - K-means için XGBoost
X_train_kmeans, X_test_kmeans, y_train_kmeans, y_test_kmeans = train_test_split(numeric_data, kmeans_labels, test_size=0.2, random_state=5)

# Ölçeklendirme
scaler = StandardScaler()
X_train_kmeans_scaled = scaler.fit_transform(X_train_kmeans)
X_test_kmeans_scaled = scaler.transform(X_test_kmeans)

# XGBoost sınıflandırma modeli - K-means
classifier_kmeans = xgb.XGBClassifier()
classifier_kmeans.fit(X_train_kmeans_scaled, y_train_kmeans)
y_pred_kmeans = classifier_kmeans.predict(X_test_kmeans_scaled)

# Sınıflandırma (Classification) - DBSCAN için XGBoost
X_train_dbscan, X_test_dbscan, y_train_dbscan, y_test_dbscan = train_test_split(numeric_data, dbscan_labels, test_size=0.2, random_state=5)

# Ölçeklendirme
X_train_dbscan_scaled = scaler.fit_transform(X_train_dbscan)
X_test_dbscan_scaled = scaler.transform(X_test_dbscan)

# XGBoost sınıflandırma modeli - DBSCAN
classifier_dbscan = xgb.XGBClassifier()
classifier_dbscan.fit(X_train_dbscan_scaled, y_train_dbscan)
y_pred_dbscan = classifier_dbscan.predict(X_test_dbscan_scaled)

# Sonuçları Değerlendirme
accuracy_kmeans = accuracy_score(y_test_kmeans, y_pred_kmeans)
f1_kmeans = f1_score(y_test_kmeans, y_pred_kmeans, average='weighted')

accuracy_dbscan = accuracy_score(y_test_dbscan, y_pred_dbscan)
f1_dbscan = f1_score(y_test_dbscan, y_pred_dbscan, average='weighted')

print("K-means Clustering Results:")
print("XGBoost Classifier Accuracy:", accuracy_kmeans)
print("XGBoost Classifier F1 Score:", f1_kmeans)

print("DBSCAN Clustering Results:")
print("XGBoost Classifier Accuracy:", accuracy_dbscan)
print("XGBoost Classifier F1 Score:", f1_dbscan)

print("K-means Clustering Labels:", kmeans_labels)
print("DBSCAN Clustering Labels:", dbscan_labels)
