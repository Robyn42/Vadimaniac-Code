import numpy as np
import pandas as pd
from model_config import OBS_LEN, PRED_LEN
import os
import tensorflow as tf


def file_gen(dir):
    """
    Generator for all csv file names in a directory.
    """
    for file in os.listdir(dir):
        yield os.path.join(dir, file)


def load_data(mode="social", batch_size=1000, dir ='../../features/val/'):
    """
    Generate data batch by batch.

    Usage:
    for batch in load_data(batch_size=1000):
        model.call(batch_data)

    The returned numpy array with three axis, (batch size, time step, feature length)

    NOTE: Currently the file locations are hardcoded. This can be updated in 
    later revisions.
    """
    if mode == 'social':
        input_features = {
            'position_x': 6,
            'position_y': 7,
            "MIN_DISTANCE_FRONT": 17,
            "MIN_DISTANCE_BACK": 18,
            "NUM_NEIGHBORS": 19,
        }
    else:
        input_features = {
            'position_x': 6,
            'position_y': 7
        }

    batch_size = batch_size
    batch_data = np.zeros((batch_size, OBS_LEN + PRED_LEN, len(input_features)))

    files = file_gen(dir)
    i = 0
    for file in files:
        batch_data[i % batch_size] = pd.read_csv(file,
                                                 usecols=input_features.keys()).values
        i += 1
        if i % batch_size == 0:
            yield batch_data
            batch_data = np.zeros((batch_size, OBS_LEN + PRED_LEN, len(input_features)))

        if i >= 1000:
            break
    if i % batch_size != 0:
        yield batch_data


if __name__ == '__main__':
    gen = load_data()
    # Print first scenario in first two batches
    print(next(gen)[:1])
    print(next(gen)[:1])
