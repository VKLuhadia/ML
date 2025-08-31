
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

def make_data(n_samples=500, n_features=15):
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    return X

def main():
    X = make_data()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    pca = PCA(n_components=0.95).fit(Xs)
    print("PCA components for 95% variance:", pca.n_components_)

    kpca = KernelPCA(n_components=5, kernel="rbf", gamma=0.1)
    X_kpca = kpca.fit_transform(Xs)
    print("Kernel PCA projected shape:", X_kpca.shape)

if __name__ == "__main__":
    main()
