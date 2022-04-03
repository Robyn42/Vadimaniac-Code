
# Vadimaniac-Code
Repository for Vadimaniacs' Heresy's project code

I added the create_venv.sh we use for class along with an updated requirements.txt.
The requirements.txt adds the pyarrow library to a virtual environment. 
Pyarrow is used to open the parquet and feather files in the Argoverse2 download. 

The preprocess.py may be usable with many different types of models.

The mlp_model_1.py model is a very basic 2 layer mlp that predicts 5 of the features that were
chosen in the preprocess.py code. The only visualization that it has at this time is 
an output of the testing loss.

Switching the data input into the model to an '5-gram' layout greatly improved the 
testing loss. Currently, the loss is often between 100 and 400 with the number of epochs 
tested being 100 and 1000. The model takes much longer to train per epoch. Currently,
with the limited dataset, it takes roughly 45 min for 1000 epochs. 


The gru_model.py model has one GRU layer with two dense layers. There is a 
visualization for the batch loss on this one. The training loss decreases 
very smoothly before 1000 epochs. There may be overfitting though since the 
test loss is fairly high to start but stabalizes at ~2500 after 1000 epochs as well.
