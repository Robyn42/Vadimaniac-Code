from typing import Dict, Tuple
import pandas as pd
import numpy as np

feature_params = {'obs_len': 20, # Observed length of the trajectory
                  "pred_len": 30 # Prediction Horizon
                  }

def compute_features(
        seq_path: str,
        social_features_utils_instance: SocialFeaturesUtils,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute social and map features for the sequence.

    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        merged_features (numpy array): SEQ_LEN x NUM_FEATURES
    """
    # TODO: AV2: start_timestamp, end_timestamp
    df = pd.read_parquet(seq_path, dtype={"TIMESTAMP": str})

    # Get social and map features for the agent
    agent_track = df[df["track_id"] == 'AV'].values

    # Social features are computed using only the observed trajectory
    social_features = compute_social_features(
        df, agent_track, feature_params.obs_len, feature_params.obs_len + feature_params.pred_len,
        RAW_DATA_FORMAT)

    # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
    # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
    if agent_track.shape[0] == feature_params.obs_len:
        agent_track_seq = np.full(
            (feature_params.obs_len + feature_params.pred_len, agent_track.shape[1]), None)
        agent_track_seq[:feature_params.obs_len] = agent_track
        merged_features = np.concatenate((agent_track_seq, social_features), axis=1)
    else:
        merged_features = np.concatenate(
            (agent_track, social_features), axis=1)
    return merged_features
