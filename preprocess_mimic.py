# preprocess_mimic.py
import pandas as pd
import numpy as np
from mimic_extract import simple_imputer, WINDOW_SIZE, GAP_TIME


def preprocess():
    # Load original data
    statics = pd.read_hdf("data/all_hourly_data.h5", "patients")
    data_full_lvl2 = pd.read_hdf("data/all_hourly_data.h5", "vitals_labs")

    # Filter patients
    statics = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME]

    # Prepare targets
    Ys = statics.loc[:, ["mort_hosp", "mort_icu", "los_icu"]]
    Ys.loc[:, "mort_hosp"] = (Ys.loc[:, "mort_hosp"]).astype(int)
    Ys.loc[:, "mort_icu"] = (Ys.loc[:, "mort_icu"]).astype(int)
    Ys.loc[:, "los_3"] = (Ys.loc[:, "los_icu"] > 3).astype(int)
    Ys.loc[:, "los_7"] = (Ys.loc[:, "los_icu"] > 7).astype(int)
    Ys.drop(columns=["los_icu"], inplace=True)

    # Process time series data
    lvl2 = data_full_lvl2.loc[
        (
            data_full_lvl2.index.get_level_values("icustay_id").isin(
                set(Ys.index.get_level_values("icustay_id"))
            )
        )
        & (data_full_lvl2.index.get_level_values("hours_in") < WINDOW_SIZE),
        :,
    ]

    # Normalize
    idx = pd.IndexSlice
    lvl2_means = lvl2.loc[:, idx[:, "mean"]].mean(axis=0)
    lvl2_stds = lvl2.loc[:, idx[:, "mean"]].std(axis=0)
    vals_centered = lvl2.loc[:, idx[:, "mean"]] - lvl2_means
    lvl2.loc[:, idx[:, "mean"]] = vals_centered

    # Impute and pivot
    lvl2 = simple_imputer(lvl2)
    lvl2_flat = lvl2.pivot_table(
        index=["subject_id", "hadm_id", "icustay_id"], columns=["hours_in"]
    )

    # Save preprocessed data
    np.save("data/mimic_X.npy", lvl2_flat.values)
    np.save("data/mimic_Y.npy", Ys.values)
    pd.Series(Ys.index.get_level_values("subject_id")).to_pickle(
        "data/mimic_subjects.pkl"
    )


if __name__ == "__main__":
    preprocess()
