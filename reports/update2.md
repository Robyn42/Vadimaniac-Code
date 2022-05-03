## Project Reflection

### Introduction

>This is a time-series regression task. We are training and assessing the robustness of deep learning models on the Argoverse 2 motion prediction dataset. Our team has developed multiple model architectures to date including stacked LSTM, multi-module GRU in an observe/predict configuration, single module GRU, and multi-layer perceptron. Additionally, our team is interested in exploration of adversarial attacks on these models. Given that this technology has been used for self-driving cars on public roads, there are serious ethical and technical implications that may follow from the brittleness of such models. Our team’s choice of dataset and approach to experimentation hopes to speak to these issues. 

### Challenges: 

>What has been the hardest part of the project you’ve encountered so far?

The hardest part we encountered was making the prediction/observation model, first understanding how it is conceptually different from the language model, the sequence to sequence model, and autoencoders, and to come to the conclusion that it may need a GRU cell. Afterwards, it was also difficult to reason about and debug the different inputs and their dimensionalities to the GRU and the GRU cell.

### Insights: 

>Are there any concrete results you can show at this point? How is your model performing compared with expectations?

The insights below are based on some of the preliminary observations of the various models. There is additional fine-tuning to follow. With our preliminary research in mind, each of the model architectures exhibits different strengths when applied to the motion forecasting data we’re using. For example, the single module GRU reaches as low a level of training loss as the multi-layer perceptron, but it requires more epochs to do so. The situation reverses when considering the prediction results; the single GRU model’s prediction accuracy on chosen inference data is seemingly higher.

In addition, we saw that the observer/prediction based GRU model showed a similar level of training loss with the single module GRU module during the initial epochs. However, these losses dropped down significantly after we changed the learning rate from 0.01 to 0.1, demonstrating that certain minor modifications that address potential numerical instabilities, for instance aligning input value magnitude with the magnitude of the learning rates, will very likely have a large impact.

The multi-layer perceptron benefits greatly in the way the data is prepared for the model and the way the prediction task is framed. For example, testing loss decreased significantly when the model was trained using “ground truth” values that are the difference between timesteps as opposed to directly plugging in values. 

### Plan: 

>Are you on track with your project?

We are at a stage where we can fine-tune our models, add small model features, and run them over our large dataset. We have not yet started to adversarially attack models, as we have been focusing on developing and improving the models themselves. 

What do you need to dedicate more time to?

Besides our main tasks to look to fine-tune our models and adversarially attack them, we may dedicate more time to visualize our results on figures that show the vehicle’s x, y positions, for both model improvement and presentation purposes.

What are you thinking of changing, if anything?

1. Try different normalization techniques. We believe that large x, y numbers make the GRU models too slow to train. Numerically, the neural networks may converge too slowly to the large x and y positional values.

2. To address the same problem as above, we saw some promising results using large learning rates. It could be worthwhile to experiment with scheduling a large learning rate in the very beginning of the training, as an alternative or an addition to normalization techniques.
