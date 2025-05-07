import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression as Iso
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class CaliForestTweak(ClassifierMixin, BaseEstimator):
    """
    A calibrated random forest classifier that combines decision trees with post-hoc calibration.

    This classifier trains an ensemble of decision trees and applies either isotonic regression
    or logistic regression calibration to improve probability estimates.

    Parameters:
        n_estimators (int): Number of trees in the forest (default: 300).
        criterion (str): Splitting criterion ("gini" or "entropy") (default: "gini").
        max_depth (int): Maximum depth of the trees (default: 5).
        min_samples_split (int): Minimum samples required to split an internal node (default: 2).
        min_samples_leaf (int): Minimum samples required to be at a leaf node (default: 1).
        ctype (str): Calibration type ("isotonic" or "logistic") (default: "isotonic").
        alpha0 (float): Prior parameter for calibration weights (default: 100).
        beta0 (float): Prior parameter for calibration weights (default: 25).
        use_oob_weight (bool): Whether to use out-of-bag weights (default: True).
    """

    def __init__(
        self,
        n_estimators=300,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        ctype="isotonic",
        alpha0=100,
        beta0=25,
        use_oob_weight=True,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.use_oob_weight = use_oob_weight

    def fit(self, X, y):
        """
        Fit the calibrated random forest model.

        Args:
            X (array-like): Training input samples.
            y (array-like): Target values (binary).

        Returns:
            self: Returns an instance of self.
        """
        # Validate input data
        X, y = check_X_y(X, y, accept_sparse=False)

        # Initialize estimators and calibrator
        self.estimators = []
        self.calibrator = None

        # Create decision tree estimators
        for i in range(self.n_estimators):
            self.estimators.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    # Note: Using sqrt for feature subset selection
                    # max_features="auto" was deprecated at newer version
                    # for DecisionTreeClassifier
                    max_features="sqrt",  
                )
            )

        # Initialize the appropriate calibrator
        if self.ctype == "logistic":
            self.calibrator = LR(penalty=None, solver="saga", max_iter=5000)
        elif self.ctype == "isotonic":
            self.calibrator = Iso(y_min=0, y_max=1, out_of_bounds="clip")

        n, m = X.shape
        # Initialize arrays for out-of-bag predictions
        Y_oob = np.full((n, self.n_estimators), np.nan)
        n_oob = np.zeros(n)
        IB = np.zeros((n, self.n_estimators), dtype=int)
        OOB = np.full((n, self.n_estimators), True)

        # Generate bootstrap indices
        for eid in range(self.n_estimators):
            IB[:, eid] = np.random.choice(n, n)
            OOB[IB[:, eid], eid] = False

        # Train each estimator and collect out-of-bag predictions
        for eid, est in enumerate(self.estimators):
            ib_idx = IB[:, eid]  # In-bag indices
            oob_idx = OOB[:, eid]  # Out-of-bag indices
            est.fit(X[ib_idx, :], y[ib_idx])
            Y_oob[oob_idx, eid] = est.predict_proba(X[oob_idx, :])[:, 1]
            n_oob[oob_idx] += 1

        # Filter samples with sufficient out-of-bag predictions
        oob_idx = n_oob > 1
        Y_oob_ = Y_oob[oob_idx, :]
        n_oob_ = n_oob[oob_idx]
        z_hat = np.nanmean(Y_oob_, axis=1)  # Mean prediction for each sample
        z_true = y[oob_idx]  # True labels for calibration

        # Calculate calibration weights
        if self.use_oob_weight:
            beta = self.beta0 + np.nanvar(Y_oob_, axis=1) * n_oob_ / 2
            alpha = self.alpha0 + n_oob_ / 2
            z_weight = alpha / beta
        else:
            z_weight = np.ones(len(z_true))  # Unit weights when OOB weighting disabled

        # Fit the calibrator
        if self.ctype == "logistic":
            self.calibrator.fit(z_hat[:, np.newaxis], z_true, z_weight)
        elif self.ctype == "isotonic":
            self.calibrator.fit(z_hat, z_true, z_weight)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X (array-like): Input samples.

        Returns:
            array: Array of shape (n_samples, 2) with class probabilities.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n, m = X.shape
        n_est = len(self.estimators)
        z = np.zeros(n)
        y_mat = np.zeros((n, 2))

        # Aggregate predictions from all estimators
        for eid, est in enumerate(self.estimators):
            z += est.predict_proba(X)[:, 1]
        z /= n_est  # Average prediction

        # Apply calibration
        if self.ctype == "logistic":
            y_mat[:, 1] = self.calibrator.predict_proba(z[:, np.newaxis])[:, 1]
        elif self.ctype == "isotonic":
            y_mat[:, 1] = self.calibrator.predict(z)

        y_mat[:, 0] = 1 - y_mat[:, 1]  # Probability of class 0
        return y_mat

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X (array-like): Input samples.

        Returns:
            array: Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
