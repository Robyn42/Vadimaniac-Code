# Vadimaniac-Code
Repository for Vadimaniacs' Heresy's project code

I added the create_venv.sh we use for class along with an updated requirements.txt.
The requirements.txt adds the pyarrow library to a virtual environment. 
Pyarrow is used to open the parquet and feather files in the Argoverse2 download. 

The preprocess.py may be usable with many different types of models.

The mlp_model_1.py model is a very basic 2 layer mlp that predicts 5 of the features that were
chosen in the preprocess.py code. The only visualization that it has at this time is 
an output of the testing loss.
As it turns out, the testing loss is extremely high righ now. The primary reason may be the 
fact that it is limited to the amount of data it is seeing and that I have not done any
tweaking to the hyperparameters. 
Another reason may be the way it prepares the data. I plan to experiment with a 
version that uses the "n-gram" format.
