import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mimic_extract import extract

X_train, X_test, y_train, y_test = extract(0, "mort_hosp")

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
