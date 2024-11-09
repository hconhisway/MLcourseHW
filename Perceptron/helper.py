import numpy as np
import pandas as pd


def load_data(train_file, test_file):
    # Load training and testing data
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)

    # Separate features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Convert labels to -1 and 1
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    return X_train, y_train, X_test, y_test


def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)
