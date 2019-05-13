# Multiclass Cuisine Detection Using RNN In Tensorflow

## Architecture of RNN

The GRU (gated recurrent units) cell is adopted in the optimized RNN. The number of units for each cell is 100, the time steps are 85, and the size of the input is 50. The output of the last time step is fed into a dense layer to get the logits, which further go through the softmax activation function to get the 20 probabilities for each class. The highest probability corresponds to the predicted class. The cross entropy is used to compute the loss and the Adam optimizer is used to optimize the parameter of the model. 

<img width="40%" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/42804316/57659200-7853df80-75af-11e9-986c-a9a574eb965d.png"> <img width="55%" alt="simplified_flowchart" src="https://user-images.githubusercontent.com/42804316/57659217-873a9200-75af-11e9-9160-34fa736c1a5e.png"><br /> <center>Fig. Tensorboard Computational Graph (left) and Final Network Architecture (right)</center>

