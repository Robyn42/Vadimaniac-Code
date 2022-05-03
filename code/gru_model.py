import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tqdm import tqdm
from functools import reduce
import sys
import os
import datetime
import matplotlib.pyplot as plt
from preprocess import motion_forecasting_get_data

class GRU_Forecasting_Model(tf.keras.Model):
    def __init__(self):

        '''
        The Model is a GRU version of a RNN for predicting
        the next set of variables for a timestep from the 
        Argoverse 2 motion-forecasting dataset.

        The values that are predicted are continuous
        variables so MSE is used for the loss 
        calculation.
        '''
        super(GRU_Forecasting_Model, self).__init__()
        #super(Model, self).__init__()

        # Initialize the hyperparameters of the model.
        self.batch_size = 100
        self.window_size = 10
        self.dense_size = 100
        self.output_size = 8 # The number of features per timestep.
        self.dropout_rate = 3e-2
        self.learning_rate = 1e-3
        self.leaky_relu_alpha = 3e-1
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha = self.leaky_relu_alpha)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError(name = "MSE Loss")

        # Initialize model layers.
        self.GRU = tf.keras.layers.GRU(units = self.window_size, return_sequences = True, return_state = True)
        self.dense_1 = tf.keras.layers.Dense(self.dense_size, activation = self.leaky_relu)
        # No activation on the output layer.
        self.dense_2 = tf.keras.layers.Dense(self.output_size)
        # Using activation defined on the layer above not seperate.
        #self.relu = tf.keras.layers.LeakyReLU(alpha = self.leaky_relu_alpha)
        self.Dropout = tf.keras.layers.Dropout(rate = self.dropout_rate)

    def call(self, inputs, initial_state, model_testing = False):
        '''
        The Model's forward propagation function.
    
        param inputs: The inputs were batched before the call. Shaped (batch_size, window_size)
        param initial_state: shape (batch_size, rnn_size)
        param testing: Used to control dropout layer usage.
 
        return: predicted values for 'position_x', 'position_y', 'heading',
        'velocity_x', and 'velocity_y' and the final_state of the GRU module.
        '''

        #embed_lookup = tf.nn.embedding_lookup(self.embedding, inputs, name = "Embedding_Layer")

        if model_testing == True:
            # No dropout layer for testing.
            output_1 = self.GRU(inputs = inputs, initial_state = initial_state)
            output_2 = self.dense_1(output_1[0])
            #output_2 = self.relu(output_2)
            predictions = self.dense_2(output_2)
            # Taking the second output returned from the GRU layer as the 
            # final state.
            final_state = output_1[1]
        
        else:
            output_1 = self.GRU(inputs = inputs, initial_state = initial_state)
            output_2 = self.dense_1(output_1[0])
            #output_2 = self.relu(output_2)
            output_2 = self.Dropout(output_2)
            predictions = self.dense_2(output_2)
            # Taking the second output returned from the GRU layer as the 
            # final state.
            final_state = output_1[1]

        return predictions, final_state


    def loss(self, true_values, predictions):
        '''
        Calculates the average loss per batch after forward pass through 
        the model.

        param predictions: matrix of model predictions with 
        shape (batch_size, output_size)
        '''
    
        loss = self.loss(true_values, predictions)

        # I believe that the keras loss layer used already
        # computes the reduced value. This is here if needed.
        # Calculate the average loss for the batch
        #loss = tf.reduce_mean(loss)

        #print(f'Loss per batch: {loss}')
        return loss

def train(model, train_inputs, train_true_values):
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

    # Remove entries from the end of the input and true_value arrays
    # so that they are divisible by the window_size.
    # Using the python modulus operation to determine the 
    # number of entries that should be left off at the end.
    train_timesteps_removed = train_inputs.shape[0]%model.window_size
    
    train_inputs = train_inputs[:-train_timesteps_removed]
    train_true_values = train_true_values[:-train_timesteps_removed]

    #print(train_inputs.shape)
    #print(train_true_values.shape)
    # Reshape inputs and true_values by window_size. The final 
    # shape should match (batch_size, window_size, output_size)
    train_inputs = tf.reshape(train_inputs, [(train_inputs.shape[0] // model.window_size), model.window_size, model.output_size]) 
    train_true_values = tf.reshape(train_true_values, [(train_true_values.shape[0] // model.window_size), model.window_size, model.output_size])

    # Array to hold batch loss values
    batch_loss = np.empty(shape = [train_inputs.shape[0]//model.batch_size])

    # Set the initial_state of the model for the first run
    # of the call in the for loop below.
    initial_state = None

    # Batching method by dividing the number of input rows by the batch size.
    for i in range(train_inputs.shape[0]//model.batch_size):
        # The code here moves between the batches in the loop by 
        # shifting over by the number of batches.
        training_inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #print(training_input.shape)
        training_true_values = train_true_values[i*model.batch_size:(i+1)*model.batch_size]
        # Gradient tape scope
        with tf.GradientTape() as tape:
            forward_pass = model.call(training_inputs, initial_state, model_testing = False)
            # The output of the model needs to be reshaped as above to match the 
            # training_values before going into the loss calculation.
            forward_pass_pred = tf.reshape(forward_pass[0], [model.batch_size, model.window_size, model.output_size])
            loss = model.loss(training_true_values, forward_pass_pred)
        # Optimize
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Collect loss for the batch. 
        batch_loss[i] = loss
        # Update the initial_state for the next batch with the 
        # final_state of the previous one.
        initial_state = forward_pass[1]

    #Calculate loss over all training examples
    training_loss = tf.reduce_mean(batch_loss)

    return training_loss




def test(model, test_inputs, test_true_values):
    '''
    Tests the model over one epoch.
    The testing data is batched and the loss per batch is collected to
    calculate the loss over the entire testing set.

    params and return as above.
    '''
    # Drop timesteps that are beyond the window_size.
    # Same as in train function.
    test_timesteps_removed = test_inputs.shape[0]%model.window_size
    
    test_inputs = test_inputs[:-test_timesteps_removed]
    test_true_values = test_true_values[:-test_timesteps_removed] 

    # Array to hold batch loss values
    batch_loss = np.empty(shape = [test_inputs.shape[0]//model.batch_size])

    # Reshape inputs and true_values by window size
    test_inputs = tf.reshape(test_inputs, [(test_inputs.shape[0] // model.window_size), model.window_size, model.output_size])
    test_true_values = tf.reshape(test_true_values, [(test_true_values.shape[0] // model.window_size), model.window_size, model.output_size])

    # Set the initial_state of the model for the first run
    # of the call in the for loop below.
    initial_state = None

    # Batching method by dividing the number of input rows by the batch size.
    for i in range(test_inputs.shape[0]//model.batch_size):
        # The code here moves between the batches in the loop by 
        # shifting over by the number of batches.
        testing_inputs = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #print(training_input.shape)
        testing_true_values = test_true_values[i*model.batch_size:(i+1)*model.batch_size]

        forward_pass = model.call(testing_inputs, initial_state, model_testing = True)
        forward_pass_pred = tf.reshape(forward_pass[0], [model.batch_size, model.window_size, model.output_size])
        batch_loss[i] = model.loss(testing_true_values, forward_pass_pred)

    #Calculate loss over all testing examples
    testing_loss = tf.reduce_mean(batch_loss)

    return testing_loss


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

def results_logging(epochs, losses, training_loss, testing_loss, prediction_inputs, prediction):
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
    if os.path.exists('gru_model.log'):
        with open('gru_model.log', 'a') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            #log.write('\n' f'Loss for each epoch: {losses}')
            log.write('\n' f'Mean Training loss: {training_loss}')
            log.write('\n' f'Mean Testing loss: {testing_loss}')
            log.write('\n' f'These are the timesteps given to the model for inference prediction:')
            log.write('\n' f'{prediction_inputs}')
            log.write('\n' f'This is the predicted next timestep values:')
            log.write('\n' f'{prediction}')
            log.write('\n')
            log.write(f'-'*80)
    else:
        # If log file does not exist, create it.
        with open('gru_model.log', 'w') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            #log.write('\n' f'Loss for each epoch: {losses}')
            log.write('\n' f'Mean Training loss: {training_loss}')
            log.write('\n' f'Mean Testing loss: {testing_loss}')
            log.write('\n' f'These are the timesteps given to the model for inference prediction:')
            log.write('\n' f'{prediction_inputs}')
            log.write('\n' f'This is the predicted next timestep values:')
            log.write('\n' f'{prediction}')
            log.write('\n')
            log.write(f'-'*80)

    return None

def main():
    '''
    Main function.
    
    The data is currently composed of 11 second sequences at 10Hz so there are 110 timesteps
    per sequence.

    The model is attempting to predict the next set of features (position_x, position_y, etc.)
    at each timestep.

    The last timestep of the "x" or "inputs" data needs to be removed. Additionally, the first element of the 
    "y" or "true_values" needs to be removed as well. 
    '''

    number_timesteps = 110
    epochs = 3000

    arg_options = ['train_model', 'load_weights']
    if len(sys.argv) !=2 or sys.argv[1] not in arg_options:
        print('Usage is :python3 gru_model.py [train_model or load_weights]')
        exit()


    # The script_state is if the model should be retrained or instead the weights loaded.
    script_state = sys.argv[1]

    if script_state == 'train_model': 
        # Load data using preprocess function.
        print('Loading data...')
        #data = motion_forecasting_get_data()
        # Data is now introduced to the model as three arrays for train, validation and test.
        train_data, validation_data, test_data = motion_forecasting_get_data()

        # The data is in the shape (number of examples, number of features).
    
        print('Preparing data...')
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
        #train_inputs = train_inputs[:-1]
        #test_inputs = test_inputs[:-1]
        # This also removes the 'timestep' feature column.
        train_inputs = train_data[:-1, 1:]
        test_inputs = test_data[:-1, 1:]
        #print(train_inputs.shape)
        #print(test_inputs.shape)
        # Remove the first timestep of the true_values
        #train_true_values = train_true_values[1:]
        #test_true_values = test_true_values[1:]
        # This also removes the 'timestep' feature column.
        train_true_values = train_data[1:, 1:]
        test_true_values = test_data[1:, 1:]

        # Initialize the model
        model = GRU_Forecasting_Model()

        # Train model for a number of epochs.
        print('Model training ...')
 
        # List to hold loss values per epoch
        losses = []

        for i in tqdm(range(epochs)):
            losses.append(train(model, train_inputs, train_true_values))

        visualize_loss(losses)
        training_loss = tf.reduce_mean(losses)

        # Test model. Print the average testing loss.
        print('Model testing ...')
        #print(f"The model's average testing loss is: {test(model, test_inputs, test_true_values)}")
        testing_loss = test(model, test_inputs, test_true_values)
        print(f"The model's average testing loss is: {testing_loss}")

        # Save model weights
        print("Saving weights model...")
        #tf.saved_model.save(model, "./saved_GRU_Forecasting_Model_weights")
        model.save_weights("./saved_GRU_Forecasting_Model_weights/GRU_weights", overwrite=True)
        print("Model weights saved!")
    
    else:
        print('Loading model weights...')
        model = GRU_Forecasting_Model()
        model.load_weights("./saved_GRU_Forecasting_Model_weights/GRU_weights")
        print('Model weights loaded...')
        train_data, validation_data, test_data = motion_forecasting_get_data()
        epochs = 'None'
        losses = 'None'
        training_loss = 'None'
        testing_loss = 'None'
        

    # Select a sequence from the data for prediction.
    # Again not including the "timesteps" column as in the dataset above.
    inference_data = test_data[:10, 1:]
    # Flatten timesteps as above
    #inference_dim = inference_data.shape[0] * inference_data.shape[1]
    prediction_inputs = np.reshape(inference_data, (1, inference_data.shape[0], inference_data.shape[1]))
    #Print input timesteps
    print("Giving the model the following timesteps:")
    print(prediction_inputs)

    # Print predicted timestep 
    print("The next timestep values will be:")
    prediction = model(prediction_inputs, initial_state = None)[0]
    print(prediction)

    # Log model information and results
    #training_loss = tf.reduce_mean(losses)
    results_logging(epochs, losses, training_loss, testing_loss, prediction_inputs, prediction)

    # Save model weights
    #print("Saving model...")
    #tf.saved_model.save(model, "./saved_GRU_Forecasting_Model_weights")
    #model.save_weights("./saved_GRU_Forecasting_Model_weights/GRU_weights", overwrite=False)
    #print("Model saved!")

    pass


if __name__ == '__main__':
    main()

