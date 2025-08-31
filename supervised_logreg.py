
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_classification_synthetic(n_samples=1000, n_features=10, n_classes=2):
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples).astype(np.int64)
    return X, y

def main():
    X, y = make_classification_synthetic()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
