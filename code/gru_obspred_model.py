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
from util.visualize import visualize_trajs
import time, os

class GRU_ObsPred_Model(tf.keras.Model):
    def __init__(self, units=128, recursive_tail=True):

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
        self.units = units
        self.dense_size = 100
        self.pred_len = PRED_LEN
        self.output_size = 2  # The number of features per timestep.
        self.dropout_rate = 3e-2
        self.learning_rate = 1e-3
        self.leaky_relu_alpha = 3e-1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError(name="MSE Loss")
        self.recursive_tail = recursive_tail

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
        if self.recursive_tail:
            pred_input = inputs[:, -1, :2]
        else:
            # Squeeze only 1st axis in case the batch size is 1.
            # pred_input = tf.squeeze(obs_seq[:, -1, :], axis=1)
            pred_input = obs_seq[:, -1, :]

        pred_state = obs_final_state
        for di in range(self.pred_len):
        #     print("pred input shape", pred_input.shape)
        #     print("pred hidden state shape", pred_state.shape)
            pred_output, pred_state = self.predictor_cell(pred_input, pred_state)
            if self.recursive_tail:
                pred_output = self.predictor_tail(pred_output)
            pred_outputs.append(pred_output)
            pred_input = pred_output

            # pred_outputs = tf.concat((pred_outputs, pred_output), axis=1)

        if not self.recursive_tail:
            pred_outputs = [self.predictor_tail(pred_output) for pred_output in pred_outputs]
        pred_outputs = tf.stack(pred_outputs, axis=1)
        # print(pred_outputs.shape)
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

def train(model, data_loader, log_dst):
    """
    Trains the model over the number of epochs specified.

    data loader: a Python function that generates data batch by batch, shaped (batch_size, number of timesteps,number of features)
    return: loss per batch as an array
    """
    batch_loss = []
    batch_ades = []
    mse = []

    run_times = []
    start_time = time.time()

    num_batches = 0
    # Print header for loss reporting
    print_and_log(f"batch number, batch loss, batch ade, batch mse, runtime in s", dst=log_dst)

    for i, batch_data in enumerate(load_data(batch_size=model.batch_size, dir ='../../features/train/')):
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

        preds_abs = undo_deltas(preds.numpy(), pred_reference)
        training_truth_abs = undo_deltas(training_truth, pred_reference)
        # print_and_log(np.max(training_truth_abs - batch_data_original[:, -PRED_LEN:, :2]), dst=log_dst)
        ade = np.mean([get_ade(preds_abs[i], training_truth_abs[i]) for i in range(model.batch_size)])
        batch_ades.append(ade)
        batch_mse = np.mean(((preds_abs - training_truth_abs)**2).mean(axis=(1, 2)))
        mse.append(((preds_abs - training_truth_abs)**2).mean(axis=(1, 2)))

        curr_time = time.time()
        run_time = curr_time - start_time
        print_and_log(f"{i}, {loss}, {ade}, {batch_mse}, {run_time}", log_dst)
        start_time = curr_time
        run_times.append(run_time)

        num_batches += 1


    print_and_log("Training average metrics: loss, ade, mse", log_dst)
    print_and_log(f"{np.mean(batch_loss)}, {np.mean(batch_ades)}, {np.mean(mse)}", log_dst)
    print_and_log(f"Total samples in epoch: {model.batch_size * num_batches}",  dst=log_dst)
    print_and_log(f"Total runtime in mins: {np.sum(run_times) / 60}", log_dst)

    #Calculate loss over all training examples
    training_loss = tf.reduce_mean(batch_loss)
    return training_loss


def test(model, load_data, log_dst):
    """
    Tests the model over one epoch.
    """
    batch_loss = []
    batch_ades = []
    mse = []

    for i, batch_data in enumerate(load_data(batch_size=model.batch_size, dir='../../features/val/')):
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

        # print_and_log(f"batch {i} loss {loss}")
        preds_abs = undo_deltas(preds.numpy(), pred_reference)
        testing_truth_abs = undo_deltas(testing_truth, pred_reference)
        # print_and_log(np.max(testing_truth_abs - batch_data_original[:, -PRED_LEN:, :2]))
        ade = np.mean([get_ade(preds_abs[i], testing_truth_abs[i]) for i in range(model.batch_size)])
        batch_ades.append(ade)
        mse.append(((preds_abs - testing_truth_abs) ** 2).mean(axis=(1, 2)))

    print_and_log("Testing average metrics: loss, ade, mse", log_dst)
    print_and_log(f"{np.mean(batch_loss)}, {np.mean(batch_ades)}, {np.mean(mse)}", log_dst)
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

def print_and_log(line, dst):
    print(line)
    with open(dst, 'a') as log:
        log.write(f'{line}\n')

def run(unit_size=128):
    """
    Main function.
    
    The data is currently composed of 11 second sequences at 10Hz so there are 110 timesteps
    per sequence.

    The model is attempting to predict the next set of features (position_x, position_y, etc.)
    at each timestep.

    The last timestep of the "x" or "inputs" data needs to be removed. Additionally, the first element of the 
    "y" or "true_values" needs to be removed as well. 
    """

    continue_training = False
    # Number for the first epoch this training session starts at.
    starting_epoch = 0  # READ the log and set this as the last epoch there!!! Zero-indexed!!!
    epochs = 2
    unit_size = 256
    saved_model = False
    rec_tail = True

    if unit_size != 128:
        results_dir = f"./experiments/units/{unit_size}"
    elif rec_tail:
        results_dir = f"./experiments/pred_len/{PRED_LEN}_{OBS_LEN}"
    else:
        results_dir = f"./experiments/pred_len_flat_tail/{PRED_LEN}_{OBS_LEN}"
    if not os.path.exists(results_dir):
        print("making", results_dir)
        os.mkdir(results_dir)
    saved_weights_path = f"{results_dir}/weights/curr/"
    saved_weights_per_epoch = f"{results_dir}/weights/per_epoch/"
    log_dst = f"{results_dir}/pred_{PRED_LEN}_{unit_size}_log"

    model = GRU_ObsPred_Model(units=unit_size, recursive_tail=rec_tail)
    if saved_model == False:
        print_and_log(f"Start training at epoch={starting_epoch}...", dst=log_dst)

        if continue_training:
            model.load_weights(saved_weights_path)
            print_and_log(f"Loading weights at {saved_weights_path}t o continue training", dst=log_dst)


        # Train model for a number of epochs.
        print_and_log('Model training ...', dst=log_dst)

        start_time = time.time()
        # List to hold loss values per epoch
        losses = []
        for i in range(starting_epoch, starting_epoch + epochs):
            losses.append(train(model, load_data, log_dst=log_dst))
            model.save_weights(saved_weights_path)
            curr_time = time.time()
            print_and_log(f"Epoch {i} took {(curr_time - start_time)/60} min", dst=log_dst)
            start_time = curr_time

            print_and_log("Saving model...", dst=log_dst)
            model.save_weights(saved_weights_path)
            model.save_weights(f"{saved_weights_per_epoch}/{i}")
            print_and_log("Model saved!", dst=log_dst)
        # visualize_loss(losses)
    else:
        print_and_log(f"Testing session only...", dst=log_dst)
        model.load_weights(saved_weights_path)

    test(model, load_data, log_dst=log_dst)

    inference_data = next(load_data(batch_size=1, dir='../../features/val/'))
    pred_ref = get_reference(inference_data, -PRED_LEN)
    whole_seq_ref, whole_seq_deltas = get_deltas(inference_data)
    pred_inputs = whole_seq_deltas[:, :-PRED_LEN, :]

    pred_deltas = model(pred_inputs).numpy()
    pred_outputs = undo_deltas(pred_deltas, pred_ref)

    pred_inputs = undo_deltas(pred_inputs, whole_seq_ref)
    truth = undo_deltas(inference_data[:, -PRED_LEN:, :], pred_ref)

    # print(f"Input sequence: {pred_inputs[0, :, :2]}")
    # print(f"Ground truth (left two columns) vs prediction (right two columns): {np.hstack((truth[0, :, :2], pred_outputs[0, :, :2]))}")

    # visualize_trajs(np.squeeze(pred_inputs), np.squeeze(pred_outputs),
    #                 np.squeeze(truth), path="obs_pred_sample_result.png")
    # print_and_log(pred_outputs, dst=log_dst)

    val_data_gen = load_data(batch_size=1, dir='../../features/val/')
    for j in range(5):
        inference_data = next(val_data_gen)
        pred_ref = get_reference(inference_data, -PRED_LEN)
        whole_seq_ref, whole_seq_deltas = get_deltas(inference_data)
        pred_inputs = whole_seq_deltas[:, :-PRED_LEN, :]

        pred_deltas = model(pred_inputs).numpy()
        pred_outputs = undo_deltas(pred_deltas, pred_ref)

        pred_inputs = undo_deltas(pred_inputs, whole_seq_ref)
        truth = undo_deltas(inference_data[:, -PRED_LEN:, :], pred_ref)

        visualize_trajs(np.squeeze(pred_inputs), np.squeeze(pred_outputs),
                        np.squeeze(truth), path=f"{results_dir}/obs_pred_sample_result_{j}_e{starting_epoch + epochs - 1}.png")
    model.summary()

if __name__ == '__main__':
    for unit_size in [8, 16, 32, 512, 1024, 2048, 4096, 4096*2, 4096*4]:
        run(unit_size)
