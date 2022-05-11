# Features we use for each mode
INPUT_FEATURES = {
    "social":
    ["position_x", "position_y", "MIN_DISTANCE_FRONT", "MIN_DISTANCE_BACK", "NUM_NEIGHBORS"],
    "none": ["position_x", "position_y"],
}

OUTPUT_FEATURES = {
    "social": ["position_x", "position_y"],
    "none": ["position_x", "position_y"],
}


FULL_LEN = 110 # Length of full dataset
# OBS_LEN, PRED_LEN = 80, 30 # Baseline
# OBS_LEN, PRED_LEN = 44, 66  # 2:3 ratio between observe and predict
# OBS_LEN, PRED_LEN = 55, 55 # Same as the Lyft dataset
OBS_LEN, PRED_LEN = 20, 30 # 2:3 ratio between observe and predict