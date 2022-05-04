"""
Normalization steps we are specifically developing for our task.

Recommended first step is to generate relative distances. This step ensures that the distance values smaller.
"""
import copy
import numpy as np

DELTA_RANGE_CSV = "../delta_range_summary.csv"


def get_reference(features, reference_index=0):
    """
    Return the features array at a timestep for reference.
    """
    return copy.deepcopy(features[:, reference_index, :2])


def get_deltas(features, inplace=True):
    """
    Features is a (batch_size, time step length, number of features) numpy array.

    We assume that the first two features are x, y positions.
    """
    reference = get_reference(features)
    if not inplace:
        features = copy.deepcopy(features)
    for i in range(features.shape[1] - 1, 0, -1):
        features[:, i, :2] = features[:, i, :2] - features[:, i - 1, :2]
    features[:, 0, :2] = 0

    return reference, features

# def save_deltas_range(deltas):
#     """
#     Save the deltas range to a common array
#     """
#     ranges = np.fromtxt(DELTA_RANGE_CSV, delimiter=",")
#     mins = deltas.min(axis=(0, 1))
#     deltas.max(axis=(0, 1))
#     np.savetxt(DELTA_RANGE_CSV, a, delimiter=",")


def undo_deltas(deltas, reference, inplace=True):
    """
    Take a numpy array of deltas, and a numpy array of references returned by get_deltas. Generate
    a numpy array of features.
    """
    if not inplace:
        features = copy.deepcopy(deltas)
    else:
        features = deltas
    features[:, 0, :2] = copy.deepcopy(reference)
    for i in range(1, features.shape[1]):
        features[:, i, :2] = features[:, i, :2] + features[:, i - 1, :2]
    return features


def test_normalization():
    from load_data import load_data
    batch_data = next(load_data(batch_size=5))
    reference, deltas = get_deltas(batch_data)
    undone_batch_data = undo_deltas(deltas, reference)

    max_dis = np.max(batch_data[:, :, :2] - undone_batch_data[:, :, :2])
    print("Original data and data undone from delta calculations " +
          f"are different by at most {max_dis}.")

if __name__ == '__main__':
    test_normalization()
