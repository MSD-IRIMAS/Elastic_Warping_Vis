import numpy as np
import os

from sklearn.preprocessing import LabelEncoder


def create_directory(directory_path):

    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def znormalisation(x):

    stds = np.std(x, axis=1, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


def encode_labels(y):

    labenc = LabelEncoder()

    return labenc.fit_transform(y)


def dtw_path_to_plot(path_dtw):

    axis_x = []
    axis_y = []

    for pair in path_dtw:

        axis_x.append(pair[0])
        axis_y.append(pair[1])

    return np.asarray(axis_x), np.asarray(axis_y)
