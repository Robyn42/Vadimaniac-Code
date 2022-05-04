import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
from load_data import load_data
from model_config import OBS_LEN, PRED_LEN
from av1_features_eval.eval_forecasting import get_ade
from normalize import get_reference, get_deltas, undo_deltas


class GRU_ObsPred_Model(tf.keras.Model):
    def __init__(self):

        """
        The Model is a GRU version of an obsever/predictor model from the
        Argoverse 2 motion-forecasting dataset.

        The obersver observes a certain number of time steps, and hand the hidden
        state to the predictor.

        The predictor takes last-step observation and the hidden state, and
        reuse the last step's prediction and hidden state to make the next prediction.

        The values that are predicted are continuous
        variables so MSE is used for the loss 
        calculation.
        """
        super(GRU_ObsPred_Model, self).__init__()

        # Initialize the hyperparameters of the model.
        self.batch_size = 128
        self.units = 128
        self.dense_size = 100
        self.pred_len = PRED_LEN
        self.output_size = 2  # The number of features per timestep.
        self.dropout_rate = 3e-2
        self.learning_rate = 1e-3
        self.leaky_relu_alpha = 3e-1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError(name="MSE Loss")

        # Initialize model layers.
        # Observer inputs include both locational and social features
        self.observer = tf.keras.layers.GRU(units=self.units,
                                            return_sequences=True, return_state=True)
        # Predictor inputs have only position_x and position_y
        self.predictor_cell = tf.keras.layers.GRUCell(units=self.units)

        # Classify predictor output units into positonal outputs
        pred_dense_1 = tf.keras.layers.Dense(self.dense_size, activation='ReLU')
        pred_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        pred_dense_2 = tf.keras.layers.Dense(self.output_size)
        self.predictor_tail = tf.keras.Sequential([pred_dense_1, pred_dropout, pred_dense_2])

    def call(self, inputs, model_testing=False):
        """
        The Model's forward propagation function.
    
        param inputs: The inputs were batched before the call. Shaped (batch_size, window_size)
        param testing: Used to control dropout layer usage.
 
        return: predicted values for 'position_x', 'position_y'.
        """
        obs_seq, obs_final_state = self.observer(inputs=inputs, initial_state=None)

        # pred_outputs = tf.zeros((self.batch_size, 0, self.output))
        pred_outputs = []
        pred_input = inputs[:, -1, :2]
        pred_state = obs_final_state
        for di in range(self.pred_len):
        #     print("pred input shape", pred_input.shape)
        #     print("pred hidden state shape", pred_state.shape)
            pred_output, pred_state = self.predictor_cell(pred_input, pred_state)
            pred_output = self.predictor_tail(pred_output)
            pred_outputs.append(pred_output)
            pred_input = pred_output
            # pred_outputs = tf.concat((pred_outputs, pred_output), axis=1)

        pred_outputs = tf.stack(pred_outputs, axis=1)
        return pred_outputs


    def loss(self, true_values, predictions):
        """
        Calculates the average loss per batch after forward pass through 
        the model.

        param predictions: matrix of model predictions with 
        shape (batch_size, output_size)
        """
    
        loss = self.loss(true_values, predictions)

        # I believe that the keras loss layer used already
        # computes the reduced value. This is here if needed.
        # Calculate the average loss for the batch
        #loss = tf.reduce_mean(loss)

        #print(f'Loss per batch: {loss}')
        return loss

def train(model, data_loader):
    """
    Trains the model over the number of epochs specified.

    param data loader: a Python function that generates data batch by batch, shaped (batch_size, number of timesteps,number of features)
    return: loss per batch as an array
    """
    batch_loss = []
    batch_ades = []
    mse = []

    for i, batch_data in enumerate(load_data(batch_size=model.batch_size)):
        pred_reference = get_reference(batch_data, -PRED_LEN)
        batch_reference, batch_data = get_deltas(batch_data, inplace=True)

        training_inputs = batch_data[:, :OBS_LEN, :]
        # Retain only the target two features, position_x and position_y for true labels
        training_truth = batch_data[:, -PRED_LEN:, :2]
        # Gradient tape scope
        with tf.GradientTape() as tape:
            preds = model.call(training_inputs)
            loss = model.loss(training_truth, preds)
        # Optimize
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Collect loss for the batch.
        batch_loss.append(loss)

        # print(f"batch {i} loss {loss}")
        preds_abs = undo_deltas(preds.numpy(), pred_reference)
        training_truth_abs = undo_deltas(training_truth, pred_reference)
        # print(np.max(training_truth_abs - batch_data_original[:, -PRED_LEN:, :2]))
        ade = np.mean([get_ade(preds_abs[i], training_truth_abs[i]) for i in range(model.batch_size)])
        batch_ades.append(ade)
        mse.append(((preds_abs - training_truth_abs)**2).mean(axis=(1, 2)))

    print("average loss", np.mean(batch_loss))
    print("average ade", np.mean(batch_ades))
    print("average mse", np.mean(mse))
    #Calculate loss over all training examples
    training_loss = tf.reduce_mean(batch_loss)
    return training_loss


def test(model, test_inputs, load_data):
    """
    Tests the model over one epoch.
    """
    batch_loss = []
    batch_ades = []
    mse = []

    for i, batch_data in enumerate(load_data(batch_size=model.batch_size)):
        pred_reference = get_reference(batch_data, -PRED_LEN)
        batch_reference, batch_data = get_deltas(batch_data, inplace=True)

        testing_inputs = batch_data[:, :OBS_LEN, :]
        # Retain only the target two features, position_x and position_y for true labels
        testing_truth = batch_data[:, -PRED_LEN:, :2]
        # Gradient tape scope
        preds = model.call(testing_inputs)
        loss = model.loss(testing_truth, preds)
        # Collect loss for the batch.
        batch_loss.append(loss)

        # print(f"batch {i} loss {loss}")
        preds_abs = undo_deltas(preds.numpy(), pred_reference)
        testing_truth_abs = undo_deltas(testing_truth, pred_reference)
        # print(np.max(testing_truth_abs - batch_data_original[:, -PRED_LEN:, :2]))
        ade = np.mean([get_ade(preds_abs[i], testing_truth_abs[i]) for i in range(model.batch_size)])
        batch_ades.append(ade)
        mse.append(((preds_abs - testing_truth_abs) ** 2).mean(axis=(1, 2)))

    print("average loss", np.mean(batch_loss))
    print("average ade", np.mean(batch_ades))
    print("average mse", np.mean(mse))
    return np.mean(batch_loss)


def visualize_loss(losses):
    """
    Outputs a plot of the losses over all batches and epochs.

    params losses: array of loss values.
    return: plot
    """

    batches = np.arange(1, len(losses) + 1)
    plt.title('Loss per batch')
    plt.xlabel('batch #')
    plt.ylabel('Loss value')
    plt.plot(batches, losses)
    plt.show()

    pass


def main():
    """
    Main function.
    
    The data is currently composed of 11 second sequences at 10Hz so there are 110 timesteps
    per sequence.

    The model is attempting to predict the next set of features (position_x, position_y, etc.)
    at each timestep.

    The last timestep of the "x" or "inputs" data needs to be removed. Additionally, the first element of the 
    "y" or "true_values" needs to be removed as well. 
    """
    saved_model = False
    if saved_model == False:
        epochs = 20
        model = GRU_ObsPred_Model()

        # Train model for a number of epochs.
        print('Model training ...')
 
        # List to hold loss values per epoch
        losses = []
        for i in range(epochs):
            losses.append(train(model, load_data))
        visualize_loss(losses)

        print("Saving model...")
        tf.saved_model.save(model, f"./saved_GRU_Observe_Predict_{PRED_LEN}_Model")
        print("Model saved!")
    else:
        model = tf.saved_model.load(f"./saved_GRU_Observe_Predict_{PRED_LEN}_Model")

    inference_data = next(load_data(batch_size=1))
    pred_ref = get_reference(inference_data, -PRED_LEN)
    inference_ref, inference_deltas = get_deltas(inference_data)
    prediction_inputs = inference_deltas[:, :-PRED_LEN, :]
    #Print input timesteps
    # print("Giving the model the following timesteps:")
    # print(inference_data)

    # Print ground truth
    print(f"The next {PRED_LEN} ground truth values will be:")
    print(undo_deltas(inference_data[:, -PRED_LEN:, :], pred_ref))

    # Print predicted timesteps
    print(f"The next {PRED_LEN} predicted values will be:")
    pred_deltas = model(prediction_inputs).numpy()
    pred_outputs = undo_deltas(pred_deltas, pred_ref)
    print(pred_outputs)


if __name__ == '__main__':
    main()
