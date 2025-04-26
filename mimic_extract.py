import pandas as pd
from pandas import DataFrame, IndexSlice
import numpy as np

GAP_TIME = 6  # In hours
WINDOW_SIZE = 24  # In hours
ID_COLS = ["subject_id", "hadm_id", "icustay_id"]
TRAIN_FRAC = 0.7
TEST_FRAC = 0.3


def simple_imputer(df: DataFrame) -> DataFrame:
    """
    Performs simple imputation and feature engineering on time-series data.

    Specifically, it imputes missing 'mean' values using forward fill,
    group mean, and zero fill. It also creates a binary 'mask' indicating
    data presence and calculates 'time_since_measured' for each feature.

    Args:
        df (pd.DataFrame): Input DataFrame with a MultiIndex for columns.
                           Expected levels include feature names and aggregation
                           functions ('mean', 'count'). Rows should be indexed
                           by time and grouped by ID_COLS.

    Returns:
        pd.DataFrame: Processed DataFrame containing imputed 'mean' columns,
                      'mask' columns, and 'time_since_measured' columns.
                      Columns are sorted.
    """
    idx = IndexSlice
    df = df.copy()

    # Simplify column index if it has extra levels (e.g., from previous processing)
    if len(df.columns.names) > 2:
        df.columns = df.columns.droplevel(("label", "LEVEL1", "LEVEL2"))

    # Select only 'mean' and 'count' aggregations for processing
    df_out = df.loc[:, idx[:, ["mean", "count"]]]

    # --- Impute 'mean' values ---
    # Calculate the mean value for each feature within each ICU stay
    icustay_means = df_out.loc[:, idx[:, "mean"]].groupby(ID_COLS).mean()

    # Impute missing mean values in three steps:
    # 1. Forward fill within each group (carry last observation forward)
    # 2. Fill remaining NaNs with the pre-calculated group mean
    # 3. Fill any remaining NaNs (if a feature was missing for the entire group) with 0
    imputed_means = (
        df_out.loc[:, idx[:, "mean"]]
        .groupby(ID_COLS)
        .fillna(method="ffill")
        .groupby(ID_COLS)
        .fillna(icustay_means)
        .fillna(0)
    ).copy()
    df_out.loc[:, idx[:, "mean"]] = imputed_means

    # --- Create 'mask' feature ---
    # Create a binary mask: 1 if data was present (count > 0), 0 otherwise
    mask = (df.loc[:, idx[:, "count"]] > 0).astype(float).copy()
    # Replace original 'count' columns with the 'mask'
    df_out.loc[:, idx[:, "count"]] = mask
    # Rename the 'count' level in the column index to 'mask'
    df_out = df_out.rename(columns={"count": "mask"}, level="Aggregation Function")

    # --- Calculate 'time_since_measured' feature ---
    # 1 if the value was absent (masked), 0 otherwise
    is_absent = 1 - df_out.loc[:, idx[:, "mask"]].copy()
    # Cumulative sum of absence within each group gives total hours of absence so far
    hours_of_absence = is_absent.groupby(ID_COLS).cumsum()
    # Get the cumulative absence at the last point a measurement *was* present
    last_present_absence = (
        hours_of_absence[is_absent == 0].groupby(ID_COLS).fillna(method="ffill")
    )
    # Time since measured is the difference between total absence and absence at last measurement
    time_since_measured = hours_of_absence - last_present_absence.fillna(
        0
    )  # fillna(0) handles start of series
    # Rename the aggregation level for the new feature
    time_since_measured.rename(
        columns={"mask": "time_since_measured"},
        level="Aggregation Function",
        inplace=True,
    )

    # Add the 'time_since_measured' columns to the output DataFrame
    df_out = pd.concat((df_out, time_since_measured), axis=1)
    # If a value was never measured, fill 'time_since_measured' with a large value
    # (WINDOW_SIZE + 1 implies longer than the observation window)
    time_since_measured_filled = (
        df_out.loc[:, idx[:, "time_since_measured"]].fillna(WINDOW_SIZE + 1)
    ).copy()
    df_out.loc[:, idx[:, "time_since_measured"]] = time_since_measured_filled

    # Sort columns for consistent order
    df_out.sort_index(axis=1, inplace=True)
    return df_out


def extract(
    random_seed: int,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and preprocesses time-series and static data for model training and testing.

    This function:
        - Loads patient static and hourly time-series data from HDF5 files.
        - Filters patients with sufficient ICU stay duration.
        - Prepares target variables for prediction (mortality, length of stay, etc.).
        - Splits the data into training and testing sets by subject.
        - Normalizes the time-series features using training set statistics.
        - Applies imputation and feature engineering to handle missing values.
        - Pivots the time-series data to a flat format suitable for machine learning models.
        - Returns the processed feature arrays and target arrays for both train and test sets.

    Args:
        random_seed (int): Seed for reproducible train/test split.
        target (str): Name of the target variable to extract (e.g., 'mort_hosp', 'mort_icu', 'los_3', 'los_7').

    Returns:
        Tuple containing:
            - X_train (np.ndarray): Training set features (subjects × features × time).
            - X_test (np.ndarray): Test set features (subjects × features × time).
            - y_train (np.ndarray): Training set target values.
            - y_test (np.ndarray): Test set target values.
    """
    # Load static patient data and hourly time-series data
    statics = pd.read_hdf("data/all_hourly_data.h5", "patients")
    data_full_lvl2 = pd.read_hdf("data/all_hourly_data.h5", "vitals_labs")

    # Filter patients with ICU stays longer than the observation window + gap
    statics = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME]

    # Prepare target variables
    Ys = statics.loc[:, ["mort_hosp", "mort_icu", "los_icu"]]
    Ys.loc[:, "mort_hosp"] = (Ys.loc[:, "mort_hosp"]).astype(int)
    Ys.loc[:, "mort_icu"] = (Ys.loc[:, "mort_icu"]).astype(int)
    Ys.loc[:, "los_3"] = (Ys.loc[:, "los_icu"] > 3).astype(int)
    Ys.loc[:, "los_7"] = (Ys.loc[:, "los_icu"] > 7).astype(int)
    Ys.drop(columns=["los_icu"], inplace=True)

    # Filter time-series data to only include relevant ICU stays and time window
    lvl2 = data_full_lvl2.loc[
        (
            data_full_lvl2.index.get_level_values("icustay_id").isin(
                set(Ys.index.get_level_values("icustay_id"))
            )
        )
        & (data_full_lvl2.index.get_level_values("hours_in") < WINDOW_SIZE),
        :,
    ]

    # Ensure subject pools match between time-series and static data
    lvl2_subj_idx, Ys_subj_idx = [
        df.index.get_level_values("subject_id") for df in (lvl2, Ys)
    ]
    lvl2_subjects = set(lvl2_subj_idx)
    assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"

    # Split subjects into train and test sets
    np.random.seed(random_seed)
    subjects = np.random.permutation(list(lvl2_subjects))
    N = len(lvl2_subjects)
    N_train, N_test = int(TRAIN_FRAC * N), int(TEST_FRAC * N)
    train_subj = subjects[:N_train]
    test_subj = subjects[N_train:]

    # Split both time-series and target data by subject
    [(lvl2_train, lvl2_test), (Ys_train, Ys_test)] = [
        [
            df.loc[df.index.get_level_values("subject_id").isin(s), :]
            for s in (train_subj, test_subj)
        ]
        for df in (lvl2, Ys)
    ]

    idx = pd.IndexSlice

    # Compute mean and std for normalization using only the training set
    lvl2_means = lvl2_train.loc[:, idx[:, "mean"]].mean(axis=0)
    lvl2_stds = lvl2_train.loc[:, idx[:, "mean"]].std(axis=0)

    # Center (normalize) the 'mean' features in both train and test sets
    vals_centered = lvl2_train.loc[:, idx[:, "mean"]] - lvl2_means
    lvl2_train.loc[:, idx[:, "mean"]] = vals_centered
    vals_centered = lvl2_test.loc[:, idx[:, "mean"]] - lvl2_means
    lvl2_test.loc[:, idx[:, "mean"]] = vals_centered

    # Impute missing values and add mask/time-since-measured features
    lvl2_train, lvl2_test = [simple_imputer(df) for df in (lvl2_train, lvl2_test)]

    # Pivot the time-series data to a flat (wide) format for ML models
    lvl2_flat_train, lvl2_flat_test = [
        (
            df.pivot_table(
                index=["subject_id", "hadm_id", "icustay_id"], columns=["hours_in"]
            )
        )
        for df in (lvl2_train, lvl2_test)
    ]

    # Return features and targets for train and test sets
    return (
        lvl2_flat_train.values,
        lvl2_flat_test.values,
        Ys_train.loc[:, target].values,
        Ys_test.loc[:, target].values,
    )
