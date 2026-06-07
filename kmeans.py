from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from rfm import rfm

# 特征列
X = rfm[['Recency', 'Frequency', 'Monetary']]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means 聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

rfm['Cluster'] = kmeans.labels_

cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).round(2)

cluster_summary.rename(columns={'CustomerID': 'CustomerCount'}, inplace=True)

print("\n🏆 K-Means 客户分群结果：")
print(cluster_summary)