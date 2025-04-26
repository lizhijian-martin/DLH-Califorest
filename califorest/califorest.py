from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression as Iso
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class CaliForest(ClassifierMixin, BaseEstimator):
    """
    A calibrated random forest classifier that combines decision trees with post-hoc calibration.

    This classifier:
        - Trains an ensemble of decision trees using bootstrapped samples.
        - Uses out-of-bag (OOB) predictions to estimate calibration parameters.
        - Applies either isotonic regression or logistic regression to calibrate the predictions.
        - Supports weighted calibration to account for uncertainty in OOB predictions.

    Args:
        n_estimators (int): Number of decision trees in the ensemble.
        criterion (Literal["gini", "entropy", "log_loss"]): Splitting criterion for the trees.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum number of samples required to split a node.
        min_samples_leaf (int): Minimum number of samples required at a leaf node.
        ctype (Literal["isotonic", "logistic"]): Type of calibration to apply ("isotonic" or "logistic").
        alpha0 (int): Prior for the gamma distribution shape parameter (used in weighted calibration).
        beta0 (int): Prior for the gamma distribution scale parameter (used in weighted calibration).
    """

    def __init__(
        self,
        n_estimators: int,
        criterion: Literal["gini", "entropy", "log_loss"],
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        ctype: Literal["isotonic", "logistic"],
        alpha0: int,
        beta0: int,
    ):
        self.n_estimators = n_estimators
        self.criterion: Literal["gini", "entropy", "log_loss"] = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0

    def fit(self, X, y):
        """
        Fit the CaliForest model to the training data.

        Args:
            X (array-like): Training feature matrix of shape (n_samples, n_features).
            y (array-like): Training target vector of shape (n_samples,).

        Returns:
            self: Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.estimators = []

        # Initialize decision trees for the ensemble
        for i in range(self.n_estimators):
            self.estimators.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features="sqrt",
                )
            )

        # Initialize the calibrator based on the specified type
        if self.ctype == "logistic":
            self.calibrator = LR(penalty=None, solver="saga", max_iter=5000)
        else:
            self.calibrator = Iso(y_min=0, y_max=1, out_of_bounds="clip")

        n, m = X.shape
        Y_oob = np.full((n, self.n_estimators), np.nan)  # Store OOB predictions
        n_oob = np.zeros(n)  # Count OOB samples per observation
        IB = np.zeros((n, self.n_estimators), dtype=int)  # In-bag indices
        OOB = np.full((n, self.n_estimators), True)  # Out-of-bag mask

        # Generate bootstrap samples and OOB indices
        for eid in range(self.n_estimators):
            IB[:, eid] = np.random.choice(n, n)
            OOB[IB[:, eid], eid] = False

        # Train each tree and collect OOB predictions
        for eid, est in enumerate(self.estimators):
            ib_idx = IB[:, eid]  # In-bag indices for this tree
            oob_idx = OOB[:, eid]  # Out-of-bag indices for this tree
            est.fit(X[ib_idx, :], y[ib_idx])  # Fit the tree on in-bag samples
            Y_oob[oob_idx, eid] = est.predict_proba(X[oob_idx, :])[
                :, 1
            ]  # Store OOB predictions
            n_oob[oob_idx] += 1  # Increment OOB count for these samples

        # Filter observations with sufficient OOB predictions
        oob_idx = n_oob > 1
        Y_oob_ = Y_oob[oob_idx, :]
        n_oob_ = n_oob[oob_idx]
        z_hat = np.nanmean(Y_oob_, axis=1)  # Mean OOB prediction per observation
        z_true = y[oob_idx]  # True labels for OOB samples

        # Compute weights for calibration based on OOB variance
        beta = self.beta0 + np.nanvar(Y_oob_, axis=1) * n_oob_ / 2
        alpha = self.alpha0 + n_oob_ / 2
        z_weight = alpha / beta  # Weight for each observation

        # Fit the calibrator
        if self.ctype == "logistic":
            self.calibrator.fit(z_hat[:, np.newaxis], z_true, z_weight)
        else:
            self.calibrator.fit(z_hat, z_true, z_weight)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            array-like: Predicted probabilities of shape (n_samples, 2).
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n, m = X.shape
        n_est = len(self.estimators)
        z = np.zeros(n)  # Aggregated predictions from all trees
        y_mat = np.zeros((n, 2))  # Output probability matrix

        # Aggregate predictions from all trees
        for eid, est in enumerate(self.estimators):
            z += est.predict_proba(X)[:, 1]
        z /= n_est  # Average prediction

        # Apply calibration
        if isinstance(self.calibrator, LR):
            y_mat[:, 1] = self.calibrator.predict_proba(z[:, np.newaxis])[:, 1]
        else:
            y_mat[:, 1] = self.calibrator.predict(z)

        y_mat[:, 0] = 1 - y_mat[:, 1]  # Complement for the negative class
        return y_mat

    def predict(self, X):
        """
        Predict class labels for the input samples.

        Args:
            X (array-like): Feature matrix of shape (n_samples, n_features).

        Returns:
            array-like: Predicted class labels of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
