
# Vadimaniac-Code
Repository for Vadimaniacs' Heresy's project code

Some elements of the model and preprocess layout and interconnectivity were inspired by the 
homework and lab assignments for CS2470 - Deep Learning at Brown University. 

A create_venv.sh script is present along with an updated requirements.txt.
The requirements.txt adds the pyarrow library to a virtual environment. 
Pyarrow is used to open the parquet and feather files in the Argoverse2 download. 

The preprocess.py may be usable with many different types of models.

The three current models include visualization for loss per epoch.

There is a stacked LSTM model (lstm_model.py) that was inspired by the LSTM used as a baseline
by the Argoverse 1 team. This version uses two LSTM modules with three dense layers following.
The first LSTM module passes it's output and final state to the respective inputs of 
the second. It's log is stored as lstm_model.log.


The mlp_model_1.py model is a two layer mlp that predicts 8 of the features that were
chosen in the preprocess.py code. To use this model add either ngram_basic
or ngram_diff after mlp_model_1.py. The basic version is a '5-gram' format version. The 
'diff' version creates the true values of the '5-gram' by taking the differences 
of the 4th and 5th timestep values. 

After the model trains, it outputs a prediction for the 
values of all the features in the next timestap. There is a prechosen timeseries 
that the prediction is based on that prints just ahead of the prediction.


Logging of some of the model infomation and performance was added. The 
log file for this model is mlp_model_1.log.

Switching the data input into the model to an '5-gram' layout greatly improved the 
testing loss. I corrected an error in the processing where the data was not being properly 
flattened. Now the model trains much faster and appears to be more accurate depending on the 
number of epochs.

The 'diff' version das a much lower test loss on average. I have not noticed a 
significant difference in the quality of the predictions as of yet. There may be 
revisions needed to the prediction functionality. 

The gru_model.py model has one GRU layer with two dense layers. There is a 
visualization for the batch loss on this one. The training loss decreases 
very smoothly before 1000 epochs. There may be overfitting though since the 
test loss is fairly high to start but stabalizes after 1000 epochs as well.

This model also has logging functionality. The log name is gru_model.log.

The model presents a prediction of the next timestep just as the mlp mpdel does.
There is a difference in the way the prediction is displayed though. It appears as though some
recent revisions of the keras LSTM and GRU models do not give the user the ability 
to set the input shape. They expect batch size, timestps (or window size), feature only. Therefore,
the output is three dimensional. If we keep the number of outputs from the GRU layer at 
the window size then the timestep that is predicted all have the same value. If you increase the 
number of outputs then each timestep predicted varies by a small margin. Keeping the output 
at window size does not appear to effect the prediction accuracy but I cannot say this is the 
correct way to do this. More research is needed. 
