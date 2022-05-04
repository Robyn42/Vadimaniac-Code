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

class MLP_Forecasting_Model(tf.keras.Model):
    def __init__(self):

        '''
        The Model is a multi-layer perceptron that predicts
        the next set of variables for a timestep from the 
        Argoverse 2 motion-forecasting dataset.

        There are currently 2 dense layers with no softmax layer
        since the predictions are continuous variables.
        '''
        super(MLP_Forecasting_Model, self).__init__()
        #super(Model, self).__init__()

        # Initialize the hyperparameters of the model.
        self.batch_size = 100
        self.output_size = 8 # The number of features in the data.
        self.dropout_rate = 3e-2
        self.learning_rate = 1e-3
        self.leaky_relu_alpha = 3e-1
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha = self.leaky_relu_alpha)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError(name = "MSE Loss")

        # Initialize model layers.
        # Trying LeakyReLU instead o ReLU.
        self.dense_1 = tf.keras.layers.Dense(units = 100, activation = self.leaky_relu)
        # No activation on the layer prior to output layer.
        self.dense_2 = tf.keras.layers.Dense(units = 100)
        
        self.dense_3 = tf.keras.layers.Dense(units = self.output_size)

        self.Dropout = tf.keras.layers.Dropout(rate = self.dropout_rate)

    def call(self, inputs, testing = False):
        '''
        The Model's forward propagation function.
    
        param inputs: The inputs should have the shape (654, output_size) but this can be 
        adjusted for possible changes in model performance. The inputs
        were flattened and batched before the call.

        param testing: Used to control dropout layer usage.
 
        return: predicted values for 'position_x', 'position_y', 'heading',
        'velocity_x', 'velocity_y', along with the computed social features..
        '''

        if testing == True:
            # No dropout layer for testing.
            output_1 = self.dense_1(inputs)
            output_2 = self.dense_2(output_1)
            predictions = self.dense_3(output_2)

        else:
            x = inputs
            output_1 = self.dense_1(inputs)
            output_2 = self.Dropout(output_1)
            output_3 = self.dense_2(output_2)
            output_4 = self.Dropout(output_3)
            predictions = self.dense_3(output_4)

        return predictions


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
    Trains the model one epoch at a time. 
    The training input is batched.

    param model: The initialized model.
    param train_inputs: The flattened training inputs before batching.
    param train_true_values: Vector of true values for the last 
    timestep of each sequence.

    return: average batch loss per epoch
    '''

    # Array to hold batch loss values
    batch_loss = np.empty(shape = [train_inputs.shape[0]//model.batch_size])

    # Batching method by dividing the number of input rows by the batch size.
    for i in range(train_inputs.shape[0]//model.batch_size):
        # The code here moves between the batches in the loop by 
        # shifting over by the number of batches.
        training_inputs = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #print(training_input.shape)
        training_true_values = train_true_values[i*model.batch_size:(i+1)*model.batch_size]
        # Gradient tape scope
        with tf.GradientTape() as tape:
            forward_pass = model.call(training_inputs, testing = False)
            batch_loss = model.loss(training_true_values, forward_pass)
        # Optimize
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #Calculate loss over all training examples
    training_loss = tf.reduce_mean(batch_loss)

    return training_loss


def test(model, test_inputs, test_true_values):
    '''
    Tests the model over one epoch.
    The testing data is batched and the loss per batch is collected to
    calculate the loss over the entire testing set.

    param model: The initialized model.
    param test_inputs: The flattened testing inputs before batching.
    param train_true_values: Vector of true values for the last timestep 
    of each sequence.
    '''
    print('Model testing ...')


    # Array to hold batch loss values
    batch_loss = np.empty(shape = [test_inputs.shape[0]//model.batch_size])

    # Batching method by dividing the number of input rows by the batch size.
    for i in range(test_inputs.shape[0]//model.batch_size):
        # The code here moves between the batches in the loop by 
        # shifting over by the number of batches.
        testing_inputs = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        #print(training_input.shape)
        testing_true_values = test_true_values[i*model.batch_size:(i+1)*model.batch_size]
       
        forward_pass = model.call(testing_inputs, testing = True)
        batch_loss[i] = model.loss(testing_true_values, forward_pass)
        
    #Calculate loss over all testing examples
    testing_loss = tf.reduce_mean(batch_loss)
    
    return testing_loss

def visualize_loss(losses):
    '''
    Outputs a plot of the losses over all batches and epochs.

    params losses: array of loss values.
    return: plot
    '''

    epochs = np.arange(1, len(losses) + 1)
    plt.title('Loss per batch')
    plt.xlabel('epoch #')
    plt.ylabel('Loss value')
    plt.plot(epochs, losses)
    plt.show()

    pass

def results_logging(epochs, ngram_type_selected, losses, training_loss, testing_loss, prediction_inputs, prediction):
    '''
    Creates a log of the model's output including the number of epochs,
    the testing, and training loss.

    param epochs: The number of epochs for model run. 
    param ngram_type_selected: A 'basic' or 'difference' target value.
    param training_loss: This is the mean batch loss over epochs.
    param testing_loss: The final testing loss value.
    param prediction_inputs:
    param prediction:
    
    returns: None - information is added to existing log file.
    '''

    now = datetime.datetime.now()
    # If log file exists, append to it.
    if os.path.exists('mlp_model_1.log'):
        with open('mlp_model_1.log', 'a') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            log.write('\n' f'The n-gram format is: {ngram_type_selected}')
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
        with open('mlp_model_1.log', 'w') as log:
            log.write('\n' f'{now.strftime("%H:%M on %A, %B %d")}')
            log.write('\n' f'Number of epochs: {epochs}')
            log.write('\n' f'The n-gram format is: {ngram_type_selected}')
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

def create_ngram_diff(input):
    # Numpy arrays to hold the data.
    #inputs = np.empty(shape = [data.shape[0]//number_timesteps, (data.shape[1] - 1)*(number_timesteps - 1)]) 
    #true_values = np.empty(shape = [data.shape[0]//number_timesteps, (data.shape[1] - 1)])
    
    
    # This is the first method for creating the sequences and true values
    #for i in tqdm(range(data.shape[0]//number_timesteps)):
    #    inputs_seq, true_values_seq = data[i*(number_timesteps - 1):(i+1)*(number_timesteps - 1), 1:], data[(i+1)*(number_timesteps - 1), 1:]
    #    inputs[i] = inputs_seq.reshape(1, (data.shape[1] - 1)*(number_timesteps - 1))
    #    true_values[i] = true_values_seq

    # This is the second method for creating the sequences. It uses the concept of an ngram.
    # In this case it is an '5 gram' Where the there is 4 timesteps and you are trying to 
    # Predict the 5th. 
    # Each timestep consisting of the features (eg. position_x, position_y, heading, velocity_x, velocity_y)
    # collectively is considered to be a 'token' or 'word' and the sequence is the sentence.
    

    inputs_sequence = np.array([[input[i, 1:], input[i+1, 1:], input[i+2, 1:], input[i+3, 1:], np.subtract(input[i+3, 1:], input[i+4, 1:])] for i in range(input.shape[0] - 4)])


    # Takes the last column and splits it off the data arrays into a seperate 
    # label array.
    inputs, true_values = np.hsplit(inputs_sequence, [-1])
    #print(inputs.shape)
    #print(inputs[:10])
    #print(true_values[0])
    # The inputs need to be flattened so that dimensions 2 and 3 
    # become 1 dimension. The shape should be (the length of
    # dimension 1, dimension 2 x dimension 3)
    input_dim = inputs.shape[1] * inputs.shape[2]
    inputs = np.reshape(inputs, (inputs.shape[0], input_dim))
    true_values_dim = true_values.shape[1] * true_values.shape[2]
    true_values = np.reshape(true_values, (true_values.shape[0], true_values_dim))
    #print(inputs.shape)
    #print(true_values.shape)

    return inputs, true_values

def create_ngram_basic(input):
    # Numpy arrays to hold the data.
    #inputs = np.empty(shape = [data.shape[0]//number_timesteps, (data.shape[1] - 1)*(number_timesteps - 1)]) 
    #true_values = np.empty(shape = [data.shape[0]//number_timesteps, (data.shape[1] - 1)])
    
    
    # This is the first method for creating the sequences and true values
    #for i in tqdm(range(data.shape[0]//number_timesteps)):
    #    inputs_seq, true_values_seq = data[i*(number_timesteps - 1):(i+1)*(number_timesteps - 1), 1:], data[(i+1)*(number_timesteps - 1), 1:]
    #    inputs[i] = inputs_seq.reshape(1, (data.shape[1] - 1)*(number_timesteps - 1))
    #    true_values[i] = true_values_seq

    # This is the second method for creating the sequences. It uses the concept of an ngram.
    # In this case it is an '5 gram' Where the there is 4 timesteps and you are trying to 
    # Predict the 5th. 
    # Each timestep consisting of the features (eg. position_x, position_y, heading, velocity_x, velocity_y)
    # collectively is considered to be a 'token' or 'word' and the sequence is the sentence.
    
    inputs_sequence = np.array([[input[i, 1:], input[i+1, 1:], input[i+2, 1:], input[i+3, 1:], input[i+4, 1:]] for i in range(input.shape[0] - 4)])
    # Takes the last column and splits it off the data arrays into a seperate 
    # label array.
    inputs, true_values = np.hsplit(inputs_sequence, [-1])
    #print(inputs.shape)
    #print(inputs[:10])
    #print(true_values[0])
    # The inputs need to be flattened so that dimensions 2 and 3 
    # become 1 dimension. The shape should be (the length of
    # dimension 1, dimension 2 x dimension 3)
    input_dim = inputs.shape[1] * inputs.shape[2]
    inputs = np.reshape(inputs, (inputs.shape[0], input_dim))
    true_values_dim = true_values.shape[1] * true_values.shape[2]
    true_values = np.reshape(true_values, (true_values.shape[0], true_values_dim))
    #print(inputs.shape)
    #print(true_values.shape)

    return inputs, true_values

def main():
    '''
    Main function.
    
    param epochs: The number of epochs to train model.
    param number_timesteps: This is the number of timesteps in the sequence per prediction.
    The data is currently composed of 11 second sequences at 10Hz so there are 110 timesteps
    per sequence. So this would seperate into 109 for the input and the 110th for the 
    true_value.

    The model takes input from 3 arrays for train, validation and test. 
    This current version only used train and test at this time.
    '''

    number_timesteps = 110
    epochs = 1000

    # Choices for ngram type and training or load existing weights for the ngram
    # type selected.
    ngram_types = ['ngram_diff', 'ngram_basic', 'train_model', 'load_weights']
    if len(sys.argv) !=3 or sys.argv[1] not in ngram_types:
        print('Usage is :python3 mlp_model_1.py [ngram_basic or ngram_diff] with [train_model or load_weights]')
        exit()
    ngram_type_selected = sys.argv[1]

    # The script_state is if the model should be retrained or instead the weights loaded.
    script_state = sys.argv[2]
    if script_state == 'train_model':

        # Load data using preprocess function.
        print('Loading data...')
        #data = motion_forecasting_get_data()
        train_data, validation_data, test_data = motion_forecasting_get_data()

        # The data is in the shape (number of examples, number of features).
        # It needs to be flattened in a way that each timestep in a sequence is kept together. 
        # For each sequence the last timestep is seperated into a seperate vector to be the true_values.
        # The code below also removes the 'timestep' feature from the data since it may not have 
        # any predictive power.

        print('Preparing data...')
        # Numpy arrays to hold the data.
        #inputs = np.empty(shape = [data.shape[0]//number_timesteps, (data.shape[1] - 1)*(number_timesteps - 1)]) 
        #true_values = np.empty(shape = [data.shape[0]//number_timesteps, (data.shape[1] - 1)])
    
    
        # This is the first method for creating the sequences and true values
        #for i in tqdm(range(data.shape[0]//number_timesteps)):
        #    inputs_seq, true_values_seq = data[i*(number_timesteps - 1):(i+1)*(number_timesteps - 1), 1:], data[(i+1)*(number_timesteps - 1), 1:]
        #    inputs[i] = inputs_seq.reshape(1, (data.shape[1] - 1)*(number_timesteps - 1))
        #    true_values[i] = true_values_seq

        # This is the second method for creating the sequences. It uses the concept of an ngram.
        # In this case it is an '5 gram' Where the there is 4 timesteps and 	you are trying to 
        # Predict the 5th. 
        # Each timestep consisting of the features (eg. position_x, position_y, heading, velocity_x, velocity_y)
        # collectively is considered to be a 'token' or 'word' and the sequence is the sentence.
    
        if ngram_type_selected == 'ngram_diff':
            #inputs_sequence = np.array([[data[i, 1:], data[i+1, 1:], data[i+2, 1:], data[i+3, 1:], np.subtract(data[i+3, 1:], data[i+4, 1:])] for i in range(data.shape[0] - 4)])
            train_inputs, train_true_values = create_ngram_diff(train_data)
            test_inputs, test_true_values = create_ngram_diff(test_data)
        if ngram_type_selected == 'ngram_basic':
            #inputs_sequence = np.array([[data[i, 1:], data[i+1, 1:], data[i+2, 1:], data[i+3, 1:], data[i+4, 1:]] for i in range(data.shape[0] - 4)])
            train_inputs, train_true_values = create_ngram_basic(train_data)
            test_inputs, test_true_values = create_ngram_diff(test_data)

        # Takes the last column and splits it off the data arrays into a seperate 
        # label array.
        #inputs, true_values = np.hsplit(inputs_sequence, [-1])
        #print(inputs.shape)
        #print(inputs[:10])
        #print(true_values[0])
        # The inputs need to be flattened so that dimensions 2 and 3 
        # become 1 dimension. The shape should be (the length of
        # dimension 1, dimension 2 x dimension 3)
        #input_dim = inputs.shape[1] * inputs.shape[2]
        #inputs = np.reshape(inputs, (inputs.shape[0], input_dim))
        #true_values_dim = true_values.shape[1] * true_values.shape[2]
        #true_values = np.reshape(true_values, (true_values.shape[0], true_values_dim))
        #print(inputs.shape)
        #print(true_values.shape)

        # Split the data into train and test with roughly a 70% train, 30% test split.
        #split = np.floor(inputs.shape[0]*0.70).astype(np.int32)
        #train_inputs, train_true_values = inputs[:split, :], true_values[:split]
        #test_inputs, test_true_values = inputs[split:, :], true_values[split:]
        #print(train_inputs)
        #print(train_true_values)
    
        # Initialize model.
        model = MLP_Forecasting_Model()


        # Train model for a number of epochs.

        losses = []

        print('Model training ...')
        for i in tqdm(range(epochs)):
            losses.append(train(model, train_inputs, train_true_values))

        visualize_loss(losses)
        training_loss = tf.reduce_mean(losses)

        # Test model. Print the average testing loss.
        print('Model testing ...')
        #print(f"The model's average testing loss is: {test(model, test_inputs, test_true_values)}")
        testing_loss = test(model, test_inputs, test_true_values)
        print(f"The model's average testing loss is: {testing_loss}")

        print("Saving model weights...")
        if ngram_type_selected == 'ngram_basic':
            model.save_weights("./saved_MLP_Basic_Forecasting_Model_weights/MLP_Basic_weights", overwrite=True)
        if ngram_type_selected == 'ngram_diff':
            model.save_weights("./saved_MLP_Diff_Forecasting_Model_weights/MLP_Diff_weights", overwrite=True)
        print("Model weights saved!")

    else:
        print('Loading model weights...')
        model = MLP_Forecasting_Model()
        if ngram_type_selected == 'ngram_basic':
            model.load_weights("./saved_MLP_Basic_Forecasting_Model_weights/MLP_Basic_weights")
        if ngram_type_selected == 'ngram_diff':
            model.load_weights("./saved_MLP_Diff_Forecasting_Model_weights/MLP_Diff_weights")
        print('Model weights loaded...')
        train_data, validation_data, test_data = motion_forecasting_get_data()
        epochs = 'None'
        losses = 'None'
        training_loss = 'None'
        testing_loss = 'None'

    # Select a sequence from the data for prediction.
    # Again not including the "timesteps" column as in the dataset above.
    inference_data = test_data[:5]
    # Flatten timesteps as above
    #inference_dim = inference_data.shape[0] * inference_data.shape[1]
    if ngram_type_selected == 'ngram_basic':
       prediction_inputs, unused_values = create_ngram_basic(inference_data)
    if ngram_type_selected == 'ngram_diff':
       prediction_inputs, unused_values = create_ngram_diff(inference_data) 
    #prediction_inputs = np.reshape(inference_data, (1, inference_dim))
    #Print input timesteps
    print("These are the timesteps given to the model for inference prediction")
    print(prediction_inputs)

    # Print predicted timestep 
    print("The next timestep values will be:")
    prediction = model(prediction_inputs)
    print(prediction)

    # Log model information and results
    results_logging(epochs, ngram_type_selected, losses, training_loss, testing_loss, prediction_inputs, prediction)
    pass


if __name__ == '__main__':
    main()


