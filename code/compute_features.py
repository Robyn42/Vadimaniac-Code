from typing import Dict, Tuple
import os, time
import pandas as pd
import numpy as np
from av1_features_eval.baseline_config import RAW_DATA_FORMAT, FEATURE_FORMAT
from av1_features_eval.social_features_utils import SocialFeaturesUtils

# Parameters for use in feature computation
feature_params = {'obs_len': 80, # Observed length of the trajectory
                  # in Argverse 1, we have obs_len = 20
                  "pred_len": 30 # Prediction Horizon
                  }


def compute_features(
        seq_path: str,
        social_features_utils_instance: SocialFeaturesUtils,
) -> np.ndarray:
    """Compute social and map features for the sequence.

    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        merged_features (numpy array): SEQ_LEN x NUM_FEATURES
    """
    # TODO: verify this for AV2: start_timestamp, end_timestamp
    # df = pd.read_parquet(seq_path, dtype={"TIMESTAMP": str})
    df = pd.read_parquet(seq_path)

    # print(df[df["track_id"] == 'AV']['end_timestamp'])
    # Get social and map features for the agent
    agent_track = df[df["track_id"] == 'AV'].values
    if agent_track.shape[0] != 110:
        print(f"Skip, AV track length is {agent_track.shape[0]}")
        return None

    # Social features are computed using only the observed trajectory
    social_features = social_features_utils_instance.compute_social_features(
        df, agent_track, feature_params['obs_len'], feature_params['obs_len'] + feature_params['pred_len'],
        RAW_DATA_FORMAT)

    # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
    # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
    if agent_track.shape[0] == feature_params['obs_len']:
        agent_track_seq = np.full(
            (feature_params['obs_len'] + feature_params['pred_len'], agent_track.shape[1]), None)
        agent_track_seq[:feature_params['obs_len']] = agent_track
        merged_features = np.concatenate((agent_track_seq, social_features), axis=1)
    else:
        merged_features = np.concatenate(
            (agent_track, social_features), axis=1)
    return merged_features


def produce_files():
    """
    Produce all file names used in feature computation.
    """
    dir = '../../argoverse_2_data/motion_forecasting/val/'
    for root, dirs, files in os.walk(dir):
        for name in files:
            if os.path.splitext(name)[-1] == '.parquet':
                yield os.path.join(root, name)


def compute_and_save():
    """
    Run feature computation and save to a directory at the same level as the project code.
    """
    social_features_utils_instance = SocialFeaturesUtils()

    # Hard coded directory while each .parquet file corresponds to a scenario.
    res_dir = '../../features/val/'

    start = time.time()

    i = 0
    for fpath in produce_files():
        
        fname, _ = os.path.basename(fpath).split('.')
        merged_features = compute_features(fpath,
                                           social_features_utils_instance)
        if merged_features is None:
            continue
        df = pd.DataFrame(merged_features,
                          columns=FEATURE_FORMAT.keys())
        df.to_csv(os.path.join(res_dir, f'{fname}.csv'))

        if i > 0 and i % 1000 == 0:
            print(
                f"Feature computation for {i}th scenario completed at {(time.time() - start) / 60.0} mins"
            )
        i += 1

    print(
        f"Feature computation for validation set completed in {(time.time()-start)/60.0} mins"
    )


if __name__ == '__main__':
    compute_and_save()
