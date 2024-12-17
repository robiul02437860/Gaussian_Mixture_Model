import numpy as np
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-6, random_state=None):
        """
        Initialize the GMM model.
        :param n_components: Number of Gaussian components (K).
        :param max_iter: Maximum number of iterations for EM.
        :param tol: Convergence threshold for log-likelihood improvement.
        :param random_state: Random seed for reproducibility.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def initialize_parameters(self, X):
        """
        Randomly initialize the model parameters.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize mixing coefficients (weights)
        self.weights = np.ones(self.n_components) / self.n_components
        
        # Initialize means by sampling from data
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False,)] # randomly select n_comp samples as mean of the culters

        # Initialize covariances as identity matrices
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def e_step(self, X):
        """
        Expectation step: Compute responsibilities.
        """
        n_samples = X.shape[0]
        self.responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute responsibilities
        for k in range(self.n_components):
            self.responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
        
        # Normalize responsibilities (softmax-like behavior)
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)
    
    def m_step(self, X):
        """
        Maximization step: Update model parameters.
        """
        n_samples, n_features = X.shape
        
        # Update weights
        Nk = self.responsibilities.sum(axis=0)
        print(Nk)
        self.weights = Nk / n_samples
        
        # Update means
        self.means = np.dot(self.responsibilities.T, X) / Nk[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(
                self.responsibilities[:, k] * diff.T, diff
            ) / Nk[k]
    
    def compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the data under the current model.
        """
        log_likelihood = 0
        for k in range(self.n_components):
            log_likelihood += self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )
        return np.sum(np.log(log_likelihood))
    
    def fit(self, X):
        """
        Fit the GMM model using the EM algorithm.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.initialize_parameters(X)
        
        # EM algorithm
        log_likelihoods = []
        for i in range(self.max_iter):
            # E-Step
            self.e_step(X)
            
            # M-Step
            self.m_step(X)
            
            # Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            
            # Check for convergence
            if i > 0 and abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {i}")
                break
        
        self.log_likelihoods = log_likelihoods
        return self

    def predict_proba(self, X):
        """
        Predict probabilities (responsibilities) for each component.
        """
        self.e_step(X)
        return self.responsibilities

    def predict(self, X):
        """
        Predict the most likely component for each sample.
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)


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

gmm = GaussianMixtureModel(3, 50, random_state=42)
log_likelihoods = gmm.fit(X)

# Print results
print("Final Means:\n", gmm.means)
print("Final Covariances:\n", gmm.covariances)
print("Final Weights:", gmm.weights)

# Predict cluster labels
labels = gmm.predict(X)
print("Predicted Labels:", labels)

# Fit t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels,  cmap='viridis', alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster 1', 'Cluster 2', 'Cluster 3'], title="Clusters")
plt.title('t-SNE visualization of GMM clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Cluster')
plt.show()

