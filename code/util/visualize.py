import matplotlib.pyplot as plt

OBS_COLOR = 'blue'
TRUTH_COLOR = 'xkcd:sky blue'
PRED_COLOR = 'orange'

def visualize_trajs(obs, preds, truth, path=""):
    """
    Visualize observation, prediction, and truth arrays together

    We assume that all input arrays are of shape [time steps, any shape].
    The first entry in the second axis is x position, and the following entry the y
    position.
    """
    plt.plot(obs[:, 0], obs[:, 1], color=OBS_COLOR, label="Observation")
    plt.plot(truth[:, 0], truth[:, 1], color=TRUTH_COLOR, label="Truth")
    plt.plot(preds[:, 0], preds[:, 1], color=PRED_COLOR, label="Prediction")
    plt.legend()
    plt.title("Trajectory Prediction Results")
    if not path:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
    plt.close()

