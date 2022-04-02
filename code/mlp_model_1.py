import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers 
from tqdm import tqdm
from functools import reduce
from preprocess import motion_forecasting_get_data

class Model(tf.keras.Model):
    def __init__(self):

        '''
        The Model is a multi-layer perceptron that predicts
        the next set of variables for a timestep from the 
        Argoverse 2 motion-forecasting dataset.

        There are currently 2 dense layers with no softmax layer
        since the predictions are continuous variables.
        '''

        super(Model, self).__init__()

        # Initialize the hyperparameters of the model.
        self.batch_size = 5
        self.output_size = 5
        self.dropout_rate = 3e-2
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.loss = tf.keras.losses.MeanSquaredError(name = "MSE Loss")

        # Initialize model layers.
        # Trying LeakyReLU instead o ReLU.
        self.dense_1 = tf.keras.layers.Dense(300, activation = 'LeakyReLU')
        # No activation on this layer as the output layer.
        self.dense_2 = tf.keras.layers.Dense(self.output_size)

        self.Dropout = tf.keras.layers.Dropout(rate = self.dropout_rate)

    def call(self, inputs, testing = False):
        '''
        The Model's forward propagation function.
    
        param inputs: The inputs should have the shape (654, output_size) but this can be 
        adjusted for possible changes in model performance. The inputs
        were flattened and batched before the call.

        param testing: Used to control dropout layer usage.
 
        return: predicted values for 'position_x', 'position_y', 'heading',
        'velocity_x', and 'velocity_y'.
        '''

        if testing == True:
            # No dropout layer for testing.
            x = inputs
            output_1 = self.dense_1(x)
            predictions = self.dense_2(output_1)

        else:
            x = inputs
            output_1 = self.dense_1(x)
            output_1 = self.Dropout(output_1)
            predictions = self.dense_2(output_1)

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

def main():
    '''
    Main function.
    
    param epochs: The number of epochs to train model.
    param number_timesteps: This is the number of timesteps in the sequence per prediction.
    The data is currently composed of 11 second sequences at 10Hz so there are 110 timesteps
    per sequence. So this would seperate into 109 for the input and the 110th for the 
    true_value.
    '''

    number_timesteps = 110
    epochs = 100

    # Load data using preprocess function.
    print('Loading data...')
    data = motion_forecasting_get_data()

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
    # In this case it is an '5 gram' Where the there is 4 timesteps and you are trying to 
    # Predict the 5th. 
    # Each timestep consisting of the features (eg. position_x, position_y, heading, velocity_x, velocity_y)
    # collectively is considered to be a 'token' or 'word' and the sequence is the sentence.
    inputs_sequence = np.array([[data[i, 1:], data[i+1, 1:], data[i+2, 1:], data[i+3, 1:], data[i+4, 1:]] for i in range(data.shape[0] - 4)])

    # Takes the last column and splits it off the data arrays into a seperate 
    # label array.
    inputs, true_values = np.hsplit(inputs_sequence, [-1])

    #print(inputs[:10])
    #print(true_values[0])


    # Split the data into train and test with roughly a 70% train, 30% test split.

    split = np.floor(inputs.shape[0]*0.70).astype(np.int32)

    train_inputs, train_true_values = inputs[:split, :], true_values[:split]
    test_inputs, test_true_values = inputs[split:, :], true_values[split:]
    #print(train_inputs)
    #print(train_true_values)
    
    # Initialize model.
    model = Model()


    # Train model for a number of epochs.
    
    print('Model training ...')
    for i in tqdm(range(epochs)):
        train(model, train_inputs, train_true_values)


    # Test model. Print the average testing loss.
    print(f"The model's average testing loss is: {test(model, test_inputs, test_true_values)}")


    return

if __name__ == '__main__':
    main()


