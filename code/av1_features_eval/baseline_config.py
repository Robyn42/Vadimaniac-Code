"""This module defines all the config parameters."""

FEATURE_FORMAT = {
    'observed': 0,
    'track_id': 1,
    'object_type': 2,
    'object_category': 3,
    'timestep': 4,
    'position_x': 5,
    'position_y': 6,
    'heading': 7,
    'velocity_x': 8,
    'velocity_y': 9,
    'scenario_id': 10,
    'start_timestamp': 11,
    'end_timestamp': 12,
    'num_timestamps': 13,
    'focal_track_id': 14,
    'city': 15,
    # Below from social features computation
    "MIN_DISTANCE_FRONT": 16,
    "MIN_DISTANCE_BACK": 17,
    "NUM_NEIGHBORS": 18,
    # Below from Maps only
    # "OFFSET_FROM_CENTERLINE": 9,
    # "DISTANCE_ALONG_CENTERLINE": 10,
}

RAW_DATA_FORMAT = {
    'observed': 0,
    'track_id': 1,
    'object_type': 2,
    'object_category': 3,
    'timestep': 4,
    'position_x': 5,
    'position_y': 6,
    'heading': 7,
    'velocity_x': 8,
    'velocity_y': 9,
    'scenario_id': 10,
    'start_timestamp': 11,
    'end_timestamp': 12,
    'num_timestamps': 13,
    'focal_track_id': 14,
    'city': 15
}

LSTM_HELPER_DICT_IDX = {
    "CENTROIDS": 0,
    "CITY_NAMES": 1,
    "CANDIDATE_CENTERLINES": 2,
    "CANDIDATE_NT_DISTANCES": 3,
    "TRANSLATION": 4,
    "ROTATION": 5,
    "CANDIDATE_DELTA_REFERENCES": 6,
    "DELTA_REFERENCE": 7,
    "SEQ_PATHS": 8,
}

# Feature computation
_FEATURES_SMALL_SIZE = 100

# Social Feature computation
PADDING_TYPE = "REPEAT"  # Padding type for partial sequences
STATIONARY_THRESHOLD = (
    13)  # index of the sorted velocity to look at, to call it as stationary
VELOCITY_THRESHOLD = 1.0  # Velocity threshold for stationary
EXIST_THRESHOLD = (
    15
)  # Number of timesteps the track should exist to be considered in social context
DEFAULT_MIN_DIST_FRONT_AND_BACK = 100.0  # Default front/back distance
NEARBY_DISTANCE_THRESHOLD = 50.0  # Distance threshold to call a track as neighbor
FRONT_OR_BACK_OFFSET_THRESHOLD = 5.0  # Offset threshold from direction of travel
