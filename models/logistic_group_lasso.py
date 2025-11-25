import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class LogisticGroupLasso(BaseEstimator, ClassifierMixin):
    """
    Logistic Group Lasso classifier implemented using Proximal Gradient Descent (FISTA).
    
    Parameters
    ----------
    groups : array-like of shape (n_features,)
        Group label for each feature.
    alpha : float, default=0.01
        Regularization strength (lambda).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for stopping criteria.
    learning_rate : float, default=0.1
        Initial learning rate for backtracking line search.
    """
    def __init__(self, groups, alpha=0.01, max_iter=1000, tol=1e-4, learning_rate=0.1):
        self.groups = np.array(groups)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        n_samples, n_features = X.shape
        
        # Identify unique groups and their indices
        self.unique_groups_ = np.unique(self.groups)
        self.group_indices_ = {g: np.where(self.groups == g)[0] for g in self.unique_groups_}
        self.group_weights_ = {g: np.sqrt(len(idxs)) for g, idxs in self.group_indices_.items()}
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Optimization loop (FISTA)
        w = np.zeros(n_features + 1) # +1 for intercept
        y_k = w.copy()
        t_k = 1.0
        
        for k in range(self.max_iter):
            w_old = w.copy()
            
            # Gradient Step
            grad = self._compute_gradient(X, y, y_k)
            
            # Backtracking Line Search (simplified: just constant or decaying LR often works, 
            # but here we use a fixed step size for simplicity or simple decay)
            step_size = self.learning_rate / (1 + 0.01 * k) 
            
            w_grad = y_k - step_size * grad
            
            # Proximal Step (Group Soft Thresholding)
            w_new = self._proximal_operator(w_grad, step_size)
            
            # FISTA update
            t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            y_k = w_new + ((t_k - 1) / t_new) * (w_new - w_old)
            
            t_k = t_new
            w = w_new
            
            # Check convergence
            if np.linalg.norm(w - w_old) < self.tol:
                break
                
        self.intercept_ = w[0]
        self.coef_ = w[1:]
        
        return self
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_gradient(self, X, y, w):
        intercept = w[0]
        coef = w[1:]
        
        logits = intercept + X @ coef
        probs = self._sigmoid(logits)
        
        error = probs - y
        
        grad_intercept = np.sum(error) / len(y)
        grad_coef = (X.T @ error) / len(y)
        
        return np.concatenate(([grad_intercept], grad_coef))
    
    def _proximal_operator(self, w, step_size):
        # w[0] is intercept, not regularized
        w_new = np.zeros_like(w)
        w_new[0] = w[0] 
        
        coef = w[1:]
        
        for g in self.unique_groups_:
            idxs = self.group_indices_[g]
            w_g = coef[idxs]
            norm_g = np.linalg.norm(w_g)
            
            # Threshold
            threshold = step_size * self.alpha * self.group_weights_[g]
            
            if norm_g > threshold:
                w_new[idxs + 1] = (1 - threshold / norm_g) * w_g
            else:
                w_new[idxs + 1] = 0.0
                
        return w_new
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        logits = self.intercept_ + X @ self.coef_
        probs = self._sigmoid(logits)
        return np.vstack([1 - probs, probs]).T
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
