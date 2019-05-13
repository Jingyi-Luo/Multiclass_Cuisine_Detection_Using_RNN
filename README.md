# Multiclass Cuisine Detection Using RNN In Tensorflow

This project aims at building a RNN model with TensorFlow using a JSON-formatted dataset to detect a cuisine type from 20 classes based on the ingredients of recipes with varying length. In order to represent each word with a numeric vector, the pre-trained word embedding GloVe was used.

The data can be accessed here: [Data](https://www.kaggle.com/c/whats-cooking-kernels-only/data)

## Architecture of RNN

The GRU (gated recurrent units) cell is adopted in the optimized RNN. The number of units for each cell is 100, the time steps are 85, and the size of the input is 50. The output of the last time step is fed into a dense layer to get the logits, which further go through the softmax activation function to get the 20 probabilities for each class. The highest probability corresponds to the predicted class. The cross entropy is used to compute the loss and the Adam optimizer is used to optimize the parameter of the model. 

<img width="40%" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/42804316/57659200-7853df80-75af-11e9-986c-a9a574eb965d.png"> <img width="55%" alt="simplified_flowchart" src="https://user-images.githubusercontent.com/42804316/57659217-873a9200-75af-11e9-9160-34fa736c1a5e.png"><br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fig. Tensorboard Computational Graph (left) and Final Network Architecture (right)

**Mathematical Functions Involved**<br />
<img width="232" alt="mathematical_functions" src="https://user-images.githubusercontent.com/42804316/57660359-35e0d180-75b4-11e9-9524-89c8a71a9c62.png">


