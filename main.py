import csv
import numpy as np
import time
import os
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from califorest.califorest import CaliForest
from califorest.rc30 import RC30
from califorest.metrics import hosmer_lemeshow
from califorest.metrics import reliability
from califorest.metrics import spiegelhalter
from califorest.metrics import scaled_brier_score
import mimic_extract as mimic


def read_data(dataset, random_seed):
    """
    Load and preprocess the specified dataset for training and testing.

    Args:
        dataset (str): Name of the dataset to load. Options include:
            - "hastie": Hastie-10-2 synthetic dataset.
            - "breast_cancer": Breast cancer dataset.
            - "mimic3_mort_hosp": MIMIC-III dataset for hospital mortality.
            - "mimic3_mort_icu": MIMIC-III dataset for ICU mortality.
            - "mimic3_los_3": MIMIC-III dataset for length of stay (3-day threshold).
            - "mimic3_los_7": MIMIC-III dataset for length of stay (7-day threshold).
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) containing the training and testing data.

    Source:
        https://github.com/yubin-park/califorest/blob/master/analysis/run_chil_exp.py
    """
    X_train, X_test, y_train, y_test = None, None, None, None

    if dataset == "hastie":
        np.random.seed(random_seed)
        poly = PolynomialFeatures()
        X, y = make_hastie_10_2(n_samples=10000)
        X = poly.fit_transform(X)
        y[y < 0] = 0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    elif dataset == "breast_cancer":
        np.random.seed(random_seed)
        poly = PolynomialFeatures()
        X, y = load_breast_cancer(return_X_y=True)
        X = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    elif dataset == "mimic3_mort_hosp":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, "mort_hosp")

    elif dataset == "mimic3_mort_icu":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, "mort_icu")

    elif dataset == "mimic3_los_3":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, "los_3")

    elif dataset == "mimic3_los_7":
        X_train, X_test, y_train, y_test = mimic.extract(random_seed, "los_7")

    return X_train, X_test, y_train, y_test


def init_models(n_estimators, max_depth):
    """
    Initialize a dictionary of machine learning models for comparison.

    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.

    Returns:
        dict: Dictionary of initialized models with keys as model names and values as model instances.

    Source:
        https://github.com/yubin-park/califorest/blob/master/analysis/run_chil_exp.py
    """
    mss = 3  # Minimum samples required to split an internal node
    msl = 1  # Minimum samples required to be at a leaf node
    models = {
        "CF-Iso": CaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="isotonic",
        ),
        "CF-Logit": CaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="logistic",
        ),
        "RC-Iso": RC30(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="isotonic",
        ),
        "RC-Logit": RC30(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
            ctype="logistic",
        ),
        "RF-NoCal": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=mss,
            min_samples_leaf=msl,
        ),
    }
    return models


def run(dataset, random_seed, n_estimators=300, depth=10):
    """
    Train and evaluate models on the specified dataset.

    Args:
        dataset (str): Name of the dataset to use.
        random_seed (int): Random seed for reproducibility.
        n_estimators (int, optional): Number of trees in the forest. Defaults to 300.
        depth (int, optional): Maximum depth of the trees. Defaults to 10.

    Returns:
        list: List of results for each model, including performance metrics.

    Source:
        https://github.com/yubin-park/califorest/blob/master/analysis/run_chil_exp.py
    """
    X_train, X_test, y_train, y_test = read_data(dataset, random_seed)

    output = []

    models = init_models(n_estimators, depth)

    for name, model in models.items():
        t_start = time.time()
        model.fit(X_train, y_train)
        t_elapsed = time.time() - t_start
        y_pred = model.predict_proba(X_test)[:, 1]

        score_auc = roc_auc_score(y_test, y_pred)
        score_hl = hosmer_lemeshow(y_test, y_pred)
        score_sh = spiegelhalter(y_test, y_pred)
        score_b, score_bs = scaled_brier_score(y_test, y_pred)
        rel_small, rel_large = reliability(y_test, y_pred)

        row = [
            dataset,
            name,
            random_seed,
            score_auc,
            score_b,
            score_bs,
            score_hl,
            score_sh,
            rel_small,
            rel_large,
        ]

        print(
            ("[info] {} {}: {:.3f} sec & BS {:.5f}").format(
                dataset, name, t_elapsed, score_b
            )
        )

        output.append(row)

    return output


if __name__ == "__main__":
    # Initialize the output with column headers
    output = [
        [
            "dataset",
            "model",
            "random_seed",
            "auc",
            "brier",
            "brier_scaled",
            "hosmer_lemshow",
            "speigelhalter",
            "reliability_small",
            "reliability_large",
        ]
    ]

    # Choose one of the 6 datasets
    # dataset = "mimic3_mort_icu"
    dataset = "mimic3_los_3"  # currently doing this one
    # dataset = "mimic3_los_7"
    # dataset = "mimic3_mort_hosp" 

    # Adjust the number of estimators and depth of trees according to the paper
    n = 300  # Number of estimators
    d = 10  # Maximum depth of trees

    # Run the experiment for 10 random seeds
    for rs in range(10):
        output += run(dataset, rs, n_estimators=n, depth=d)

    # Save results to a CSV file
    fn = "results/{}.csv".format(dataset)
    
    # Ensure the results directory exists
    results_dir = os.path.dirname(fn)
    os.makedirs(results_dir, exist_ok=True)

    with open(fn, "w") as fp:
        writer = csv.writer(fp)
        writer.writerows(output)
