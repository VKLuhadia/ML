
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_blobs_synthetic(n_samples=600, centers=3, n_features=2):
    means = np.random.randn(centers, n_features).astype(np.float32) * 5
    X = []
    per = n_samples // centers
    for m in means:
        Xc = m + np.random.randn(per, n_features).astype(np.float32)
        X.append(Xc)
    return np.vstack(X)

def main():
    X = make_blobs_synthetic()
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_test)
    print("KMeans Silhouette Score:", silhouette_score(X_test, labels))

    db = DBSCAN(eps=0.5, min_samples=5).fit(X_test)
    clusters = len(set(db.labels_) - {-1})
    print("DBSCAN clusters:", clusters)

if __name__ == "__main__":
    main()
