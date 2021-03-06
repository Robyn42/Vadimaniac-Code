import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tqdm import tqdm
from functools import reduce
import sys
import os
import datetime
import time
import matplotlib.pyplot as plt
from preprocess import motion_forecasting_get_data
from load_data import load_data
from av1_features_eval.eval_forecasting import get_ade
from normalize import get_reference, get_deltas, undo_deltas
from util.visualize import visualize_trajs

class LSTM_Forecasting_Model(tf.keras.Model):
    def __init__(self):

        '''
        The Model is a LSTM version of a RNN for predicting
        the next set of variables for a timestep from the 
        Argoverse 2 motion-forecasting dataset.

        The values that are predicted are continuous
        variables so MSE is used for the loss 
        calculation.
        '''
        super(LSTM_Forecasting_Model, self).__init__()
        #super(Model, self).__init__()

        # Initialize the hyperparameters of the model.
        self.batch_size = 128
        self.window_size = 10
        self.dense_size = 100
        self.units = 128
        self.output_size = 5 # The number of output features per timestep.
        self.dropout_rate = 3e-2
        self.learning_rate = 1e-3
        self.leaky_relu_alpha = 3e-1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.mse_loss = tf.keras.losses.MeanSquaredError(name = "MSE Loss")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha = self.leaky_relu_alpha)
        # Initialize model layers.
        self.LSTM_module_1 = tf.keras.layers.LSTM(units = self.units, return_sequences = True, return_state = True)
        #self.dense_1 = tf.keras.layers.Dense(self.dense_size, activation = self.leaky_relu)
        #self.dense_2 = tf.keras.layers.Dense(self.dense_size, activation = self.leaky_relu)
        # Using activation defined on the layer above not seperate.
        #self.relu = tf.keras.layers.LeakyReLU(alpha = self.leaky_relu_alpha)
        
        self.Dropout = tf.keras.layers.Dropout(rate = self.dropout_rate)

        self.LSTM_module_2 = tf.keras.layers.LSTM(units = self.units, return_sequences = True, return_state = True)
        self.dense_1 = tf.keras.layers.Dense(self.dense_size, activation = self.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(self.dense_size, activation = self.leaky_relu)
        # No activation on the output layer.
        self.dense_3 = tf.keras.layers.Dense(self.output_size)

    def call(self, inputs, initial_state, model_testing = False):
        '''
        The Model's forward propagation function.
    
        param inputs: The inputs were batched before the call. Shaped (batch_size, window_size)
        param initial_state: shape (batch_size, rnn_size)
        param testing: Used to control dropout layer usage.
 
        return: predicted values for 'position_x', 'position_y', 'heading',
        'velocity_x', and 'velocity_y' and the final_state of the GRU module.
        '''
        
        if model_testing == True:
            # No dropout layer for testing.
            output_1 = self.LSTM_module_1(inputs = inputs, initial_state = initial_state)
            #output_2 = self.dense_1(output_1[0])
            #output_2 = self.relu(output_2)
            #output_3 = self.dense_2(output_2)
            LSTM_module_1_final_state = [output_1[1], output_1[2]]

            output_2 = self.LSTM_module_2(inputs = output_1[0], initial_state = LSTM_module_1_final_state)
            output_3 = self.dense_1(inputs = output_2[0])
            # Dense layer not in use to match other model layouts.
            #output_4 = self.dense_2(inputs = output_3) 
            predictions = self.dense_3(inputs = output_3)

            # Taking the second and third outputs returned from the LSTM layer as the 
            # final state.
            final_state = [output_2[1], output_2[2]]
        
        else:
            output_1 = self.LSTM_module_1(inputs = inputs, initial_state = initial_state)
            LSTM_module_1_final_state = [output_1[1], output_1[2]]

            output_2 = self.LSTM_module_2(inputs = output_1[0], initial_state = LSTM_module_1_final_state)
            output_3 = self.dense_1(inputs = output_2[0])
            output_4 = self.Dropout(inputs = output_3)
            # Dense layer not in use to match other model layouts.
            #output_5 = self.dense_2(inputs = output_4)
            #output_6 = self.Dropout(inputs = output_5)
            predictions = self.dense_3(output_4)
            # Taking the second and third output returned from the LSTM layer as the 
            # final state.
            final_state = [output_2[1], output_2[2]]
        # Removed the return of the final_state
        return predictions


    def loss_function(self, true_values, predictions):
        '''
        Calculates the average loss per batch after forward pass through 
        the model.

        param predictions: matrix of model predictions with 
        shape (batch_size, output_size)
        '''
         
        loss = self.mse_loss(true_values, predictions)

        # I believe that the keras loss layer used already
        # computes the reduced value. This is here if needed.
        # Calculate the average loss for the batch
        #loss = tf.reduce_mean(loss)

        #print(f'Loss per batch: {loss}')
        return loss

def train(model): #train_inputs, train_true_values):
    '''
    Trains the model over the number of epochs specified.
    The inputs and true values need to be reshaped by the window_size.
    Both are batched. Then the values returned from the model need 
    to be reshaped to (model.batch_size, model.window_size, model.output_size)
    before going into the loss calculation.

    param train_inputs: shaped (number of timesteps, number of features)
    param train_true_values: shaped (number of timesteps, number of features)
    return: loss per batch as an array
    '''
    # NOTE: Many of the lines in this section are commented out to adapt the code
    # to use the same preprocessing strategy as the gru_obspred_model.
    # It was not deleted to retain for reference.

 
    # Remove entries from the end of the input and true_value arrays
    # so that they are divisible by the window_size.
    # Using the python modulus operation to determine the 
    # number of entries that should be left off at the end.
    #train_timesteps_removed = train_inputs.shape[0]%model.window_size
    
    #train_inputs = train_inputs[:-train_timesteps_removed]
    #train_true_values = train_true_values[:-train_timesteps_removed]

    #print(train_inputs.shape)
    #print(train_true_values.shape)
    # Reshape inputs and true_values by window_size. The final 
    # shape should match (batch_size, window_size, output_size)
    #train_inputs = tf.reshape(train_inputs, [(train_inputs.shape[0] // model.window_size), model.window_size, model.output_size]) 
    #train_true_values = tf.reshape(train_true_values, [(train_true_values.shape[0] // model.window_size), model.window_size, model.output_size])

    # Array to hold batch loss values
    #batch_loss = np.empty(shape = [train_inputs.shape[0]//model.batch_size])
    batch_loss = []
    batch_ades = []
    batch_mse = []
    num_batches = 0
    start_time = time.time()
    # Set the initial_state of the model for the first run
    # of the call in the for loop below.
    initial_state = None

    # Batching method by dividing the number of input rows by the batch size.
    #for i in range(train_inputs.shape[0]//model.batch_size):
        # The code here moves between the batches in the loop by 
        # shifting over by the number of batches.
        #training_inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #print(training_input.shape)
        #training_true_values = train_true_values[i*model.batch_size:(i+1)*model.batch_size]
    for i, batch_data in enumerate(load_data(batch_size = model.batch_size, dir ='/ltmp/features/train/')):
        pred_reference = get_reference(batch_data, 0)
        batch_reference, batch_data = get_deltas(batch_data, inplace=True)

        # The last row of the inputs is deleted.
        training_inputs = batch_data[:, :-1, :]
        #training_inputs = batch_data[:, :79, :]
        #print(training_inputs)
        # The "social features" are not predicted so they are
        # not included in the loss function calculations.
        # The first row of the true values is deleted.
        training_true_values = batch_data[:, 1:, :]
        #training_true_values = batch_data[:, 1:80, :2]
        # Gradient tape scope
        with tf.GradientTape() as tape:
            forward_pass = model.call(training_inputs, initial_state, model_testing = False)
            # The output of the model needs to be reshaped as above to match the 
            # training_values before going into the loss calculation.
            #forward_pass_pred = tf.reshape(forward_pass[0], [model.batch_size, model.window_size, model.output_size])
            #loss = model.loss_function(training_true_values, forward_pass_pred)
            loss = model.loss_function(training_true_values, forward_pass)
        # Optimize
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Collect loss for the batch. 

        batch_loss.append(loss)
        preds_abs = undo_deltas(forward_pass.numpy(), pred_reference)
        training_truth_abs = undo_deltas(training_true_values, pred_reference)
        # print_and_log(np.max(training_truth_abs - batch_data_original[:, -PRED_LEN:, :2]), dst=log_dst)
        ade = np.mean([get_ade(preds_abs[i][:,:2], training_truth_abs[i][:,:2]) for i in range(model.batch_size)])
        batch_ades.append(ade)
        #batch_mse = np.mean(((preds_abs - training_truth_abs)**2).mean(axis=(1, 2)))
        batch_mse.append(((preds_abs - training_truth_abs)**2).mean(axis=(1, 2)))
        num_batches += 1
        # Update the initial_state for the next batch with the 
        # final_state of the previous one.
        #initial_state = [forward_pass[1], forward_pass[2]]

    end_time = time.time()
    run_time = end_time - start_time

    #Calculate loss over all training examples
    #training_loss = tf.reduce_mean(batch_loss)
    # Switching calculation to numpy operations in an attempt
    # to ensure the metrics match the gru_obspred_model
    training_loss = np.mean(batch_loss)
    training_ade = np.mean(batch_ades)
    training_mse = np.mean(batch_mse)
    
    return training_loss, training_ade, training_mse, num_batches, run_time


def test(model): #, test_inputs, test_true_values):
    '''
    Tests the model over one epoch.
    The testing data is batched and the loss per batch is collected to
    calculate the loss over the entire testing set.

    params and return as above.
    '''
    # NOTE: As in the train function, some of the code below is being commented 
    # out to allow for the usage of the same preprocessing functionality 
    # in use by the gru_obspred_model.

    # Drop timesteps that are beyond the window_size.
    # Same as in train function.
    #test_timesteps_removed = test_inputs.shape[0]%model.window_size
    
    #test_inputs = test_inputs[:-test_timesteps_removed]
    #test_true_values = test_true_values[:-test_timesteps_removed] 

    # Array to hold batch loss values
    #batch_loss = np.empty(shape = [test_inputs.shape[0]//model.batch_size])

    # Reshape inputs and true_values by window size
    #test_inputs = tf.reshape(test_inputs, [(test_inputs.shape[0] // model.window_size), model.window_size, model.output_size])
    #test_true_values = tf.reshape(test_true_values, [(test_true_values.shape[0] // model.window_size), model.window_size, model.output_size])

    batch_loss = []
    batch_ades = []
    batch_mse = []
    num_batches = 0
    start_time = time.time()

    # Set the initial_state of the model for the first run
    # of the call in the for loop below.
    initial_state = None

    # Batching method by dividing the number of input rows by the batch size.
    #for i in range(test_inputs.shape[0]//model.batch_size):
        # The code here moves between the batches in the loop by 
        # shifting over by the number of batches.
        #testing_inputs = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #print(training_input.shape)
        #testing_true_values = test_true_values[i*model.batch_size:(i+1)*model.batch_size]
    for i, batch_data in enumerate(load_data(batch_size = model.batch_size, dir ='/ltmp/features/val/')):
        pred_reference = get_reference(batch_data, 0)
        batch_reference, batch_data = get_deltas(batch_data, inplace=True)
        # The last row of the inputs is deleted.
        #testing_inputs = batch_data[:, :-1, :] 
        testing_inputs = batch_data[:, :79, :]
        #print(testing_inputs)
        # The "social features" are not predicted so they are
        # not included in the loss function calculations.
        # The first row of the true values is deleted.
        #testing_true_values = batch_data[:, 1:, :2]
        testing_true_values = batch_data[:, 1:80, :]
        
        forward_pass = model.call(testing_inputs, initial_state, model_testing = True)
        
        #forward_pass_pred = tf.reshape(forward_pass[0], [model.batch_size, model.window_size, model.output_size])
        # Collect loss for the batch.
        #batch_loss[i] = model.loss_function(testing_true_values, forward_pass_pred)
        loss = model.loss_function(testing_true_values, forward_pass)
        batch_loss.append(loss)
        preds_abs = undo_deltas(forward_pass.numpy(), pred_reference)
        testing_truth_abs = undo_deltas(testing_true_values, pred_reference)
        
        ade = np.mean([get_ade(preds_abs[i][:,:2], testing_truth_abs[i][:,:2]) for i in range(model.batch_size)])
        batch_ades.append(ade)
        batch_mse.append(((preds_abs - testing_truth_abs)**2).mean(axis=(1, 2)))

        num_batches += 1

    end_time = time.time()
    run_time = end_time - start_time
  
    #Calculate loss over all testing examples
    #testing_loss = tf.reduce_mean(batch_loss)
    # Switching calculation to numpy operations in an attempt
    # to ensure the metrics match the gru_obspred_model
    testing_loss = np.mean(batch_loss)
    testing_ade = np.mean(batch_ades)
    testing_mse = np.mean(batch_mse)
    #print(testing_loss)
    #print(testing_ade)
    #print(testing_mse)
    return testing_loss, testing_ade, testing_mse, num_batches, run_time 


def visualize_loss(losses):
    '''
    Outputs a plot of the losses over all batches and epochs.

    params losses: array of loss values.
    return: None - displays plot
    '''

    epochs = np.arange(1, len(losses) + 1)
    plt.title('Loss per epoch')
    plt.xlabel('epoch #')
    plt.ylabel('Loss value')
    plt.plot(epochs, losses)
    plt.show()

    return None

def results_logging(epochs, losses, ades, mses, train_runtime, test_losses, test_ades, test_mses, test_runtime):
    '''
    Creates a log of the model's output including the number of epochs,
    the testing, and training loss.

    param epochs: The number of epochs for model run.
    param training_loss: This is the mean batch loss over epochs.
    param testing_loss: The final testing loss value.
    param prediction_inputs:
    param prediction:
    
    returns: None - information is added to existing log file.
    '''

    now = datetime.datetime.now()
    # If log file exists, append to it.
    if os.path.exists('lstm_stacked_model.log'):
        with open('lstm_stacked_model.log', 'a') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            log.write('\n' f'Training loss: {losses}')
            log.write('\n' f'Training ADE: {ades}')
            log.write('\n' f'Training MSE: {mses}')
            log.write('\n' f'Training runtime: {train_runtime}')
            log.write('\n' f'Testing loss: {test_losses}')
            log.write('\n' f'Testing ADE: {test_ades}')
            log.write('\n' f'Testing MSE: {test_mses}')
            log.write('\n' f'Testing runtime: {test_runtime}')
            #log.write('\n' f'Loss for each epoch: {losses}')
            #log.write('\n' f'Mean Training loss: {training_loss}')
            #log.write('\n' f'Mean Testing loss: {testing_loss}')
            #log.write('\n' f'These are the timesteps given to the model for inference prediction:')
            #log.write('\n' f'{prediction_inputs}')
            #log.write('\n' f'This is the predicted next timestep values:')
            #log.write('\n' f'{prediction}')
            log.write('\n')
            log.write(f'-'*80)
    else:
        # If log file does not exist, create it.
        with open('lstm_stacked_model.log', 'w') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            log.write('\n' f'Training loss: {losses}')
            log.write('\n' f'Training ADE: {ades}')
            log.write('\n' f'Training MSE: {mses}')
            log.write('\n' f'Training runtime: {train_runtime}')
            log.write('\n' f'Testing loss: {test_losses}')
            log.write('\n' f'Testing ADE: {test_ades}')
            log.write('\n' f'Testing MSE: {test_mses}')
            log.write('\n' f'Testing runtime: {test_runtime}')
            #log.write('\n' f'Loss for each epoch: {losses}')
            #log.write('\n' f'Mean Training loss: {training_loss}')
            #log.write('\n' f'Mean Testing loss: {testing_loss}')
            #log.write('\n' f'These are the timesteps given to the model for inference prediction:')
            #log.write('\n' f'{prediction_inputs}')
            #log.write('\n' f'This is the predicted next timestep values:')
            #log.write('\n' f'{prediction}')
            log.write('\n')
            log.write(f'-'*80)

    return None

def prediction_function(model, inference_data):
    pred_timesteps = []
    pred_timesteps.append(inference_data)
    #pred_timesteps.append(np.reshape(inference_data, (1, inference_data.shape[0], inference_data.shape[1])))
    #social_features = np.array([[0,0,0]])
    #pred_timesteps.append(np.hstack((inference_data, social_features)))
    for i in range(30):
        prediction_inputs = pred_timesteps[i]
        #prediction_inputs = np.reshape(prediction_inputs, (1, prediction_inputs.shape[0], prediction_inputs.shape[1]))
        #pred_timesteps.append(np.hstack((np.reshape(model(prediction_inputs, initial_state = None), (1, 2)), social_features)))
        pred_timesteps.append(model(prediction_inputs, initial_state = None)) 
    return np.array(pred_timesteps)[:,:, -3]

def main():
    '''
    Main function.
    
    The data is currently composed of 11 second sequences at 10Hz so there are 110 timesteps
    per sequence. The sequences are for the main agent of the scenario resulting from our 
    computation of social features.

    The model is attempting to predict the next set of features (position_x, position_y, etc.)
    at each timestep.

    The last timestep of the "x" or "inputs" data needs to be removed. Additionally, the first element of the 
    "y" or "true_values" needs to be removed as well. 
    '''

    number_timesteps = 110
    epochs = 1

    arg_options = ['train_model', 'load_weights']
    if len(sys.argv) !=2 or sys.argv[1] not in arg_options:
        print('Usage is :python3 lstm_model.py [train_model or load_weights]')
        exit()


    # The script_state is if the model should be retrained or instead the weights loaded.
    script_state = sys.argv[1]

    if script_state == 'train_model': 
        # Load data using preprocess function.
        #print('Loading data...')
        #data = motion_forecasting_get_data()
        # Data is now introduced to the model as three arrays for train, validation and test.
        #train_data, validation_data, test_data = motion_forecasting_get_data()

        # The data is in the shape (number of examples, number of features).
    
        #print('Preparing data...')
        ### NOTE: Some code here is commented out to create compatability with 
        ### the preprocessing and data loading functions used in the gru_obspred_model.

        ## NOTE: The splitting of the data as below to create train and test data is not 
        ## Being used at this time since the preprocess now returns the 3 arrays.

        # Split the data into train and test with roughly a 70% train, 30% test split.
        # This also removes the 'timestep' feature column.
        #split = np.floor(data.shape[0]*0.70).astype(np.int32)

        #train_inputs = data[:split, 1:]
        #train_true_values = data[:split, 1:]
        #test_inputs = data[:split, 1:]
        #test_true_values = data[:split, 1:]


        # Remove the last timestep of the inputs
        #train_inputs = train_data[:-1]
        #test_inputs = validation_data[:-1]
        #print(train_inputs.shape)
        #print(test_inputs.shape)

        # Remove the first timestep of the true_values
        # This step also removes the columns for the "social features"
        # so that position_x and position_y are the only features evaluated 
        # by the loss function.
        #train_true_values = train_data[1:, :2]
        #test_true_values = validation_data[1:, :2]

        # Initialize the model
        model = LSTM_Forecasting_Model()

        # Train model for a number of epochs.
        print('LSTM Motion Forecasting Model training ...')
 
        # List to hold loss values per epoch
        losses = []
        ades = []
        mses = []

        #for i in tqdm(range(epochs)):
            #losses.append(train(model, train_inputs, train_true_values))
        training_losses, training_ades, training_mses, num_batches, training_runtime = train(model)
        losses.append(training_losses) 
        ades.append(training_ades)
        mses.append(training_mses)

        #visualize_loss(losses)
        training_loss = tf.reduce_mean(losses)
        print(losses, ades, mses, num_batches, training_runtime)
        # Test model. Print the average testing loss.
        print('LSTM Motion Forecasting Model testing ...')
        #print(f"The model's average testing loss is: {test(model, test_inputs, test_true_values)}")
        #testing_loss = test(model, test_inputs, test_true_values)
        #print(f"The model's average testing loss is: {testing_loss}")
        test_losses = []
        test_ades = []
        test_mses = []

        testing_losses, testing_ades, testing_mses, num_batches, testing_runtime = test(model)
        test_losses.append(testing_losses)
        test_ades.append(testing_ades)
        test_mses.append(testing_mses)
        testing_loss = tf.reduce_mean(test_losses)
        print(test_losses, test_ades, test_mses, num_batches, testing_runtime)
        # Save model weights
        print("Saving model...")
        #tf.saved_model.save(model, "./saved_GRU_Forecasting_Model_weights")
        model.save_weights("./saved_LSTM_Forecasting_Model_weights/LSTM_weights", overwrite=True)
        print("LSTM Model weights saved!")
    
    else:
        print('Loading model weights...')
        model = LSTM_Forecasting_Model()
        model.load_weights("./saved_LSTM_Forecasting_Model_weights/LSTM_weights")
        print('Model weights loaded...')
        #train_data, validation_data, test_data = motion_forecasting_get_data()
        epochs = 'None'
        losses = 'None'
        training_loss = 'None'
        testing_loss = 'None'
        ades = 'None'
        mses = 'None'
        test_losses = 'None'
        test_ades = 'None'
        test_mses = 'None'
        training_runtime = 'None'
        testing_runtime = 'None'
        

    # Select a sequence from the data for prediction.
    # Again not including the "timesteps" column as in the dataset above.
    #inference_data = test_data[:10, 1:]

    
    # Flatten timesteps as above
    #inference_dim = inference_data.shape[0] * inference_data.shape[1]
    #prediction_inputs = np.reshape(inference_data, (1, inference_data.shape[0], inference_data.shape[1]))
    #Print input timesteps
    #print("Giving the model the following timesteps:")
    #print(prediction_inputs)

    # Print predicted timestep 
    #print("The next timestep values will be:")
    #prediction = model(prediction_inputs, initial_state = None)[0]
    #print(prediction)

    # Log model information and results
    #training_loss = tf.reduce_mean(losses)
    results_logging(epochs, losses, ades, mses, training_runtime, test_losses, test_ades, test_mses, testing_runtime)

    val_data_gen = load_data(batch_size=1, dir='/ltmp/features/val/')
    
    inference_data = next(val_data_gen)
    pred_ref = get_reference(inference_data, -30)
    whole_seq_ref, whole_seq_deltas = get_deltas(inference_data)
    pred_inputs = whole_seq_deltas[:, :-30, :]

    pred_deltas = model(pred_inputs, initial_state = None).numpy()
    pred_outputs = undo_deltas(pred_deltas, pred_ref)
    #print(pred_outputs)
    pred_inputs = undo_deltas(pred_inputs, whole_seq_ref)
    truth = undo_deltas(inference_data[:, :-30, :], pred_ref)
    pred_outputs = prediction_function(model, pred_outputs[:,:-1,:])
    
    visualize_trajs(np.squeeze(pred_inputs), np.squeeze(pred_outputs), np.squeeze(truth))

    pass


if __name__ == '__main__':
    main()

