from typing import Tuple

import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

from aeon.datasets.tsc_datasets import univariate, multivariate
from aeon.datasets.tser_datasets import tser_soton
from aeon.datasets import load_classification, load_regression


def load_data(dataset_name: str, split: str, znormalize: bool):
    """
    Loads and preprocesses the dataset.

    Parameters
    ----------
    -dataset_name: str
        The name of the dataset to load.
    split: str:
        The data split to use (e.g., 'train', 'test').
    znormalize: bool
        Whether to apply z-normalization to the data.

    Returns:
    Tuple[np.ndarray, np.ndarray]
        The loaded data and labels.
    """
    is_classif = True

    if dataset_name in univariate or dataset_name in multivariate:
        X, y = load_classification(name=dataset_name, split=split)
    elif dataset_name in tser_soton:
        X, y = load_regression(name=dataset_name, split=split)
        is_classif = False
    else:
        raise ValueError("The dataset " + dataset_name + " does not exist in aeon.")

    if znormalize:
        X = znormalisation(x=X)
    if is_classif:
        y = encode_labels(y)

    return X, y, is_classif


def create_directory(directory_path):

    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def znormalisation(x):

    stds = np.std(x, axis=2, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=2, keepdims=True)) / stds
    return (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True))


def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)


def alignment_path_to_plot(path_dtw):

    axis_x = []
    axis_y = []

    for pair in path_dtw:

        axis_x.append(pair[0])
        axis_y.append(pair[1])

    return np.asarray(axis_x), np.asarray(axis_y)
