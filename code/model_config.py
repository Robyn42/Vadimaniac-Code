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
