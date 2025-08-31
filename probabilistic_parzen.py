
import numpy as np
import math

class ParzenWindow:
    def __init__(self, h=0.5):
        self.h = h

    def _gaussian_kernel(self, x):
        d = x.shape[1]
        norm_factor = (1.0 / ((2.0 * math.pi) ** (d / 2))) * (1.0 / (self.h ** d))
        quad = -0.5 * np.sum((x / self.h) ** 2, axis=1)
        return norm_factor * np.exp(quad)

    def fit(self, X):
        self._X = np.asarray(X, dtype=np.float32)
        return self

    def score_samples(self, X):
        logs = []
        for x in X:
            diffs = x[None, :] - self._X
            k = self._gaussian_kernel(diffs)
            density = np.mean(k)
            logs.append(np.log(density + 1e-12))
        return np.array(logs, dtype=np.float32)

def main():
    np.random.seed(0)
    X = np.random.randn(200, 2).astype(np.float32)
    model = ParzenWindow(h=0.4).fit(X)
    scores = model.score_samples(X[:5])
    print("Parzen log-densities (first 5):", scores)

if __name__ == "__main__":
    main()
