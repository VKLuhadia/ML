
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

def make_classification_synthetic(n_samples=1000, n_features=8, n_classes=3):
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples).astype(np.int64)
    return X, y

def main():
    X, y = make_classification_synthetic()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    rng = np.random.default_rng(42)
    mask = rng.random(len(y_train)) < 0.1
    y_train_semi = np.copy(y_train)
    y_train_semi[~mask] = -1

    semi = LabelSpreading(kernel="rbf", gamma=0.5, max_iter=50)
    semi.fit(X_train, y_train_semi)
    y_pred = semi.predict(X_test)
    print("Label Spreading Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
