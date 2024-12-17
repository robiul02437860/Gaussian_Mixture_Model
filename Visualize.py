import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import GaussianMixtureModel

# Generate synthetic 4D data
n_samples = 300
n_features = 4
n_components = 3

# Randomly generate synthetic clusters
np.random.seed(42)
X1 = np.random.multivariate_normal([2, 3, 5, 1], np.eye(4), size=n_samples // 3)
X2 = np.random.multivariate_normal([8, 9, 2, 3], 2 * np.eye(4), size=n_samples // 3)
X3 = np.random.multivariate_normal([5, 1, 7, 8], np.eye(4), size=n_samples // 3)
X = np.vstack([X1, X2, X3])
print(X)

gmm = GaussianMixtureModel(3, 50, random_state=42)
log_likelihoods = gmm.fit(X)

# Print results
print("Final Means:\n", gmm.means)
print("Final Covariances:\n", gmm.covariances)
print("Final Weights:", gmm.weights)

# Predict cluster labels
labels = gmm.predict(X)
print("Predicted Labels:", labels)