import numpy as np

class ELM_AE:
    def __init__(self, input_dim, hidden_dim, random_seed=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnd = np.random.RandomState(random_seed)
        self.W = self.rnd.normal(size=(input_dim, hidden_dim))
        self.b = self.rnd.uniform(-1, 1, size=(hidden_dim,))
        self.beta = None  # Will be computed in fit_transform

    def sigmoid(self, x):
        # 数值稳定的sigmoid实现，防止溢出
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def fit_transform(self, X):
        H = self.sigmoid(np.dot(X, self.W) + self.b)
        self.beta = np.linalg.pinv(H) @ X
        return H  # Encoded features

    def transform(self, X):
        if self.beta is None:
            raise ValueError("You must call fit_transform before transform.")
        H = self.sigmoid(np.dot(X, self.W) + self.b)
        return H  # Encoded features

    def reconstruct(self, X):
        if self.beta is None:
            raise ValueError("You must call fit_transform before reconstruct.")
        H = self.sigmoid(np.dot(X, self.W) + self.b)
        X_recon = H @ self.beta
        return X_recon

