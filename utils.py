import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.patches import ConnectionPatch
from matplotlib.animation import PillowWriter
import numpy as np
import os
from matplotlib.lines import Line2D

from sklearn.preprocessing import LabelEncoder
from tslearn.metrics import dtw_path
from aeon.distances import cost_matrix
from aeon.distances._alignment_paths import compute_min_return_path


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
