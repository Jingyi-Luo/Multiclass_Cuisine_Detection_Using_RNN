# Multiclass Cuisine Detection Using RNN In Tensorflow

This project aims at building a RNN model with TensorFlow using a JSON-formatted dataset to detect a cuisine type from 20 classes based on the ingredients of recipes with varying length. In order to represent each word with a numeric vector, the pre-trained word embedding GloVe was used. Accuracy and the Cohen's Kappa metric are used to measure model's performance.

The data can be accessed here: [Data](https://www.kaggle.com/c/whats-cooking-kernels-only/data)

## Architecture of RNN

The GRU (gated recurrent units) cell is adopted in the optimized RNN. The number of units for each cell is 100, the time steps are 85, and the size of the input is 50. ReLU is used as the activation function in the last time step. Then, the output is fed into a dense layer to get the logits, which further go through the softmax activation function to get the 20 probabilities for each class. The highest probability corresponds to the predicted class. The cross entropy is used to compute the loss and the Adam optimizer is used to optimize the parameter of the model. 

<img width="40%" alt="tensorboard_graph" src="https://user-images.githubusercontent.com/42804316/57659200-7853df80-75af-11e9-986c-a9a574eb965d.png"> <img width="55%" alt="simplified_flowchart" src="https://user-images.githubusercontent.com/42804316/57659217-873a9200-75af-11e9-9160-34fa736c1a5e.png"><br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fig. Tensorboard Computational Graph (left) and Simplified Flowchart (right)

The mathematical functions involved are shown below. Among them, W denotes the weights for all hidden cells and U represents the weights for all inputs x. V is the weights in the fully connected layer. Apparently, b and c are biases. It is worth mentioning that all hidden cells share the same weights W and all inputs share the same weights U across time in RNN.

**Mathematical Functions Involved**<br />
<img width="232" alt="mathematical_functions" src="https://user-images.githubusercontent.com/42804316/57660359-35e0d180-75b4-11e9-9524-89c8a71a9c62.png">

## Results

The effect of cells including basic cell, long short-term memory (LSTM) and gated recurrent unit (GRU), number of layers and dropout optimization technique have been investigated to improve the model's performance. Based on the optimized architecture which adopted one-layer GRU cell without dropout, the training set obtained the accuray of 83% and the loss of 0.53, and the test set obtained the accuracy of 75% and and the loss of 0.90 as shown below. The Cohen's Kappa was 0.72, indicating the model have favorable predictive power, rather than assigning labels randomly.

<img width="50%" alt="accuracy_train_valid" src="https://user-images.githubusercontent.com/42804316/57661467-fe285880-75b8-11e9-887b-5c4376fb5847.png"><img width="50%" alt="loss_train_valid" src="https://user-images.githubusercontent.com/42804316/57661473-08e2ed80-75b9-11e9-9598-26503d30f663.png"><br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Accuracy (left) and Loss (right) For Training and Testing Data With Epochs (graph from tensorboard)


