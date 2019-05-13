# Kaggle Competition:What is Cooking?
# Jingyi Luo

"""
A variety of cells have been investigated: Basic cell, LSTM, GRU.
"""
# ignore warning
import warnings
warnings.filterwarnings('ignore')

import os
import string
import time
import tensorflow as tf
import matplotlib.pyplot as plt
#import json
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
#import gluonnlp as nlp
from sklearn.model_selection import train_test_split

working_dir = '/Users/ljyi/Desktop/SYS6016/homework/homework_04'
os.chdir(working_dir)

# ------------------------ Data Preprocessing ---------------------------------
# read in json data using pandas
train_valid_df = pd.read_json("./data/train.json")
test_df = pd.read_json("./data/test.json")
x_train_valid_df = train_valid_df.loc[:, train_valid_df.columns != 'cuisine']
y_train_valid_df = train_valid_df.cuisine

# to download the pre-trained word embedding (glove)
#if not os.path.exists('glove.6B.zip'):
#    ! wget http://nlp.stanford.edu/data/glove.6B.zip        # download embedding
#    ! unzip glove.6B.zip                                    # unzip embedding

# function to read in the pre_trained word embedding
def load_glove_embeddings(path): # path: the path to the embedding file
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            w = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings[w] = vectors
    return embeddings
# the returned value is a big dictionary, work is key and num vector is the value
"""
the above function load_glove_embedding is from the following website:
http://androidkt.com/pre-trained-word-embedding-tensorflow-using-estimator-api/
"""

# read in embedings
embeddings = load_glove_embeddings('glove.6B.50d.txt')

# get the length of each word in the pretrained word embedding
# here we used the version with 50 numbers in a vector per word
embed_len = len(embeddings.get('i'))

# merge ingredient array and only use unique words
"""
for each list of ingredients (in one row), here is a:
.join(a): combine all the strings into one string separated by a space
.lower(): change to lower case
.split(): split the one string into a list of words based on a space
np.unique(): get the unique words for this list
"""
# a list of instances (list of list), each instance (list) contains unique words
word_list = [np.unique(' '.join(a).lower().split()) for a in x_train_valid_df.ingredients]

"""
In python, string.punctuation conatains a great list of punctuation characters as following:
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
"""
# create mapping from punctuation to none (trasparent)
# "str" is the default library from python
table = str.maketrans('', '', string.punctuation)  # type: dictionary
cleaned_list = []  # a list of lists which only contains words (cleaned list)
for i in range(len(word_list)):
    instance = word_list[i]          # a list of words ingredients for one row
    stripped = [w.translate(table) for w in instance]  # apply table to every instance to remove punctuation characters.
                                                       # only words and numbers left in stripped
    words = [word if word.isalpha() else 'num' for word in stripped] # replace number with 'num'
    cleaned_list.append(words)                         # append the cleaned instance to cleaned_list

# ========================================================
"""

# check unique words to see if existing any punctuation characters
# flatten the list of lists to a big list
flatten_cleaned_list = [item for sublist in cleaned_list for item in sublist]
# get unique words in the big list
unique_words = np.unique(flatten_cleaned_list)
# save unique words to a file for easier glance
fid = open('unique_words.txt', 'w')
for a in unique_words:
    fid.write('{0} \n'.format(a))
fid.close()

# get the length of each instance
word_count = [len(a) for a in cleaned_list]

# plot the length distribution of all the instance
bins = np.arange(0,90,2)   # 90 and 2 chosen for beauty histogram
                           # the chosen 90 need to be larger than maximum value in word_count
plt.hist(word_count, bins) # count how many elements fall into each bin and plot counted number as a bar

# check and see the longest ingredient list
index_max = np.argmax(word_count)
x_train_valid_df.ingredients[index_max]

# check and see the shortest ingredient list
index_min = np.argmin(word_count)
x_train_valid_df.ingredients[index_min]

"""

# ========================================================
"""
Apply embedings, and add zero vectors at the beginning of each instance
to match the maximum length of all instances.

"""
# get the length of each instance
word_count = [len(a) for a in cleaned_list] # cleaned_list: a list of cleaned lists
# get the maximu length
length_max = np.max(word_count)

# add zero vectors and apply embedding
converted_list = []  # a list of flattened embedding lists for all the instances
for i in range(len(cleaned_list)):
    an_instance = cleaned_list[i]    # a row
    embedded = []                    # a list of embedding lists for all the words in an instance
                                     # each word has an embedding list

    # firstly, append zero vectors to embedded
    for j in range(len(an_instance), length_max):
        embedded.append([0]*embed_len)   # embed_len: 50

    # secondly, append the word embedding to embedded
    for j in range(len(an_instance)):
        word = an_instance[j]      # get a word
        a = embeddings.get(word)   # get the word's embedding
        if a is None:
            embedded.append([0]*embed_len)   # if cannot find the word's embedding, append zero vectors
        else:
            embedded.append(list(a))         # append word embedding

    # flatten embedded to one-dimensional list
    flattened = [item for sublist in embedded for item in sublist]
    # append a list for an instance in the big list-converted list
    converted_list.append(flattened)

# convert a lit of lists to an array of array (like a matrix)
converted_list = np.array(converted_list)
# save to csv file
#converted_df.to_csv('embedded_data.csv', index=False)

# the training data
x_train_valid = converted_list
y_train_valid = y_train_valid_df.values   # .values: get array

# label encoding for y_train_valid
# label encoder: change word to number
lb = LabelEncoder()
y_train_valid = lb.fit_transform(y_train_valid)   # array
# set(y_train_valid.cuisine): {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}

# split train_valid_df into train_df and valid_df
x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.1, random_state=0)
# x_train: 35796*4250
# x_valid: 3978*4250
# y_train: 35796*1
# y_valid: 3978*1

# check class imbalance
import collections

plt.hist(y_train)
print(collections.Counter(y_train))
#Counter({9: 7038, 13: 5762, 16: 3891, 7: 2710, 3: 2397, 5: 2396, 2: 1420, 18: 1385,
#         11: 1285, 6: 1056, 17: 888, 12: 766, 19: 745, 14: 740, 1: 725, 4: 685,
#         8: 605, 10: 459, 15: 439, 0: 404})

"""
SMOTE to balance classes:
# over-sampling using SMOTE-Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
x_train_os, y_train_os = os.fit_sample(x_train, y_train)
# x_train_os: 140760*4250
# y_train_os: 140760*4250
x_train = x_train_os
y_train = y_train_os
"""

# one-hot encoding for label
y_train_encoded = np.array(pd.get_dummies(y_train))
y_valid_encoded = np.array(pd.get_dummies(y_valid))

# ---------------------------------------------------------
# test data for prediction
# # a list of instances (list of list), each instance (list) contains unique words
word_list = [np.unique(' '.join(a).lower().split()) for a in test_df.ingredients]

# clean instance
# create mapping from punctuation to none
table = str.maketrans('', '', string.punctuation)
cleaned_list = []   # a list of lists which only contains words (cleaned list)
for i in range(len(word_list)):
    instance = word_list[i]
    stripped = [w.translate(table) for w in instance]
    # change number to 'num'
    words = [word if word.isalpha() else 'num' for word in stripped]
    cleaned_list.append(words)

#  apply embedings:
converted_list = []
for i in range(len(cleaned_list)):
    an_instance = cleaned_list[i]
    embedded = []
    # add zero vectors in
    for j in range(len(an_instance), length_max):
        embedded.append([0]*embed_len)
    for j in range(len(an_instance)):
        word = an_instance[j]
        a = embeddings.get(word)
        if a is None:
            embedded.append([0]*embed_len)
        else:
            embedded.append(list(a))
    # get one-dimensional ebedding list for each ingredient list
    flattened = [item for sublist in embedded for item in sublist]
    # a list of flattened embedding lists
    converted_list.append(flattened)

# # convert a lit of lists to an array of array (like a matrix)
x_test = np.array(converted_list)

# ---------------------- Computation graph of RNN -----------------------------
tf.reset_default_graph()

n_steps = 85     # 85 time steps, n_input for each time step is 50
n_inputs = 50       # the size of the input vector
n_neurons = 100    # recurrent neurons/The number of units in the RNN cell
n_outputs = 20     # number of neurons/units of the fully connected layer
n_layers = 3

learning_rate = 0.001
batch_size = 50
n_epochs = 10

train_keep_prob = 1.0 # without dropout
#train_keep_prob = 0.5 # with dropout

keep_prob = tf.placeholder_with_default(1.0, shape=())

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
    y = tf.placeholder(tf.int32, [None, n_outputs], name="y")
#    training = tf.placeholder_with_default(False, shape=[], name='training')

# use one fully connected layer after all time steps.

# Basic RNN cell: one layer
#basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, name="Basic_RNN_Cell")
#outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# LSTMCell: one layer
#lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.relu)
#outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

# GRUCell: one layer
gru_cells = tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(gru_cells, X, dtype=tf.float32)

# Basic RNN cell: three layers without dropout
#layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
#                                            for layer in range(n_layers)]
#multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
#outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# Basic RNN cell: three layers with dropout between layers
#layers = [tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
#                                            for layer in range(n_layers)]
#layers_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in layers]
#multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers_drop)
#outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# LSTM cell: three layers with dropout
#lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.relu)
#                for layer in range(n_layers)]
#lstm_cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in lstm_cells]
#multi_layer_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells_drop)    # state_is_tuple= True (defaulted)
#outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

# GRU cell with three layers with dropout
#gru_cells = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.relu) # use_peepholes=True # activation
#                for layer in range(n_layers)] # stack three cells together to form three layers.
#gru_cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in gru_cells]
#multi_layer_cell = tf.contrib.rnn.MultiRNNCell(gru_cells_drop)    # state_is_tuple= True (defaulted)
#outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

with tf.name_scope("output"):
    # BasicRNNCell / GRUCell: one layer
    logits = tf.layers.dense(states, n_outputs, name="logits")

    # LSTMCell: one layer
#    logits = tf.layers.dense(states[1], n_outputs, name="logits")

    # BasicRNNCell /GRUCell: three layers
    # state[2] is the output of the last layer
#    logits = tf.layers.dense(states[2], n_outputs, name="logits")

    # LSTMCell: three layers
    # [2]: the 3nd layer; [1]: the hidden state (not cell state)
#    logits = tf.layers.dense(states[2][1], n_outputs, name="logits")

    Y_prob = tf.nn.softmax(logits, name="Y_prob")

# Define the optimizer; taking as input (learning_rate) and (loss)
with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate) # GradientDescentOptimizer #MomentumOptimizer , momentum=0.9
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correctPrediction = tf.equal(tf.argmax(Y_prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

#def shuffle_batch(X, y, batch_size):
#    rnd_idx = np.random.permutation(len(X))
#    n_batches = len(X) // batch_size
#    for batch_idx in np.array_split(rnd_idx, n_batches):
#        X_batch, y_batch = X[batch_idx], y[batch_idx]
#        yield X_batch, y_batch

def batch_func(X, y, batch_size):
    batches = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i: i+batch_size]
        y_batch = y[i: i+batch_size]
        mini_batch = (X_batch, y_batch)
        batches.append(mini_batch)
    return batches

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# ------------------------------ Train the model ------------------------------
# a writer to write the FFN graph Tensorboard
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# for plotting loss/accuracy plots of validation and training set on tensorboard
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./graphs/train', tf.get_default_graph())
valid_writer = tf.summary.FileWriter('./graphs/valid', tf.get_default_graph())

start = time.time()

# reshape x_valid
x_valid_reshaped = x_valid.reshape((-1, n_steps, n_inputs))

with tf.Session() as sess:
    init.run()
#    i = 0
    for epoch in range(n_epochs):
        for X_batch, y_batch in batch_func(x_train, y_train_encoded, batch_size):
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))   # need to reshape X_batch
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
#            i = i+1
#            print(i)
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch, keep_prob: train_keep_prob})
        acc_valid = accuracy.eval(feed_dict={X: x_valid_reshaped, y: y_valid_encoded})

        loss_batch = sess.run(loss, feed_dict = {X:X_batch, y:y_batch})
        loss_valid = sess.run(loss, feed_dict = {X:x_valid_reshaped, y:y_valid_encoded})
        print(epoch, "Last batch accuracy:", acc_batch, "Valid accuracy:", acc_valid)
        print(epoch, "Last batch loss:", loss_batch, "Valid loss:", loss_valid)

# run and print the outputs and states returned from RNN
#    outputs_val, states_val = sess.run([outputs, states], feed_dict = {X:X_batch})
#    print("states value:", states_val)
#    print("outputs_val:", outputs_val)

        # measure validation accuracy, and write validate summaries to FileWriters
        valid_summary, acc = sess.run([merged, accuracy], feed_dict={X: x_valid_reshaped, y: y_valid_encoded})
        valid_writer.add_summary(valid_summary, epoch)
#        print('Accuracy at step %s: %s' % (epoch, acc))

        # run training_op on training data, and add training summaries to FileWriters
        train_summary, _ = sess.run([merged, training_op], feed_dict={X:X_batch, y:y_batch}) # , training_op
        train_writer.add_summary(train_summary, epoch)

    train_writer.close()
    valid_writer.close()

    save_path = saver.save(sess, "./RNN_model.ckpt")

writer.close()

# ------------------------------ Validation  ----------------------------------
with tf.Session() as sess:
    saver.restore(sess, "./RNN_model.ckpt") # load model parameters from disk
    Z = Y_prob.eval(feed_dict = {X: x_valid_reshaped})
    y_pred = np.argmax(Z, axis = 1)   # np.argmax return the index, e.g. 0, 1, 2 for -1, 0, 1, so subtract 1.
print(pd.crosstab(y_valid, y_pred, rownames=['Actual'], colnames=['Predicted']))
print(metrics.classification_report(y_valid,y_pred))
print("Accuracy: " + str(metrics.accuracy_score(y_valid,y_pred)))
print("Cohen's Kappa: " + str(metrics.cohen_kappa_score(y_valid,y_pred)))
print('Took: %f seconds' %(time.time() - start))

# ------------------------ Prediction of test set  ----------------------------
# need to reshape X_test from (-1, 50*8) to (-1, 50, 8)
x_test_reshaped = x_test.reshape((-1, n_steps, n_inputs))

with tf.Session() as sess:
    saver.restore(sess, "./RNN_model.ckpt")
    Z = logits.eval(feed_dict = {X: x_test_reshaped})
    y_pred = np.argmax(Z, axis = 1)
print("Predicted classes:", y_pred)

# reverse label encoder: change number to word
y_pred_words = lb.inverse_transform(y_pred)

# write to a dataframe to upload to kaggle
output = pd.DataFrame(y_pred_words, columns=['cuisine'])
output['id'] = test_df.id

# swap two columns
columnsTitles=["id","cuisine"]
output=output.reindex(columns=columnsTitles)

# write to csv
output.to_csv("prediction_kaggle.csv", index=False)

# ---------------------------------- Results ----------------------------------

# 10 epochs
# -------
# BasicRNNCell: one layer
#              precision    recall  f1-score   support
#
#           0       0.32      0.11      0.16        63
#           1       0.31      0.10      0.15        79
#           2       0.51      0.60      0.55       126
#           3       0.65      0.72      0.68       276
#           4       0.33      0.23      0.27        70
#           5       0.34      0.58      0.42       250
#           6       0.43      0.44      0.43       119
#           7       0.77      0.84      0.80       293
#           8       0.25      0.23      0.24        62
#           9       0.76      0.72      0.74       800
#          10       0.59      0.33      0.42        67
#          11       0.56      0.59      0.58       138
#          12       0.46      0.41      0.43        64
#          13       0.90      0.85      0.88       676
#          14       0.56      0.59      0.58        81
#          15       0.19      0.10      0.13        50
#          16       0.52      0.66      0.58       429
#          17       0.57      0.12      0.20       101
#          18       0.67      0.61      0.64       154
#          19       0.40      0.28      0.33        80
#
#   micro avg       0.63      0.63      0.63      3978
#   macro avg       0.50      0.45      0.46      3978
#weighted avg       0.63      0.63      0.62      3978
#
#Accuracy: 0.6294620412267471
#Cohen's Kappa: 0.5861505885186096
#Took: 246.991360 seconds

# -------
# LSTMCell: one layer
#              precision    recall  f1-score   support
#
#           0       0.71      0.35      0.47        63
#           1       0.48      0.29      0.36        79
#           2       0.82      0.58      0.68       126
#           3       0.66      0.87      0.75       276
#           4       0.47      0.37      0.42        70
#           5       0.46      0.58      0.52       250
#           6       0.66      0.64      0.65       119
#           7       0.83      0.90      0.86       293
#           8       0.41      0.32      0.36        62
#           9       0.77      0.85      0.81       800
#          10       0.92      0.54      0.68        67
#          11       0.72      0.63      0.67       138
#          12       0.81      0.53      0.64        64
#          13       0.92      0.90      0.91       676
#          14       0.66      0.77      0.71        81
#          15       0.45      0.26      0.33        50
#          16       0.70      0.71      0.71       429
#          17       0.48      0.40      0.43       101
#          18       0.72      0.77      0.74       154
#          19       0.69      0.41      0.52        80
#
#   micro avg       0.73      0.73      0.73      3978
#   macro avg       0.67      0.58      0.61      3978
#weighted avg       0.73      0.73      0.72      3978
#
#Accuracy: 0.729260935143288
#Cohen's Kappa: 0.6963242979489461
#Took: 1016.985960 seconds

# -------
# GRUCell: one layer (1st run)
#              precision    recall  f1-score   support
#
#           0       0.68      0.30      0.42        63
#           1       0.42      0.27      0.33        79
#           2       0.74      0.60      0.66       126
#           3       0.75      0.84      0.79       276
#           4       0.64      0.36      0.46        70
#           5       0.52      0.56      0.54       250
#           6       0.60      0.66      0.62       119
#           7       0.80      0.89      0.84       293
#           8       0.58      0.47      0.52        62
#           9       0.72      0.90      0.80       800
#          10       0.94      0.49      0.65        67
#          11       0.70      0.72      0.71       138
#          12       0.80      0.56      0.66        64
#          13       0.90      0.91      0.90       676
#          14       0.59      0.79      0.67        81
#          15       0.45      0.26      0.33        50
#          16       0.80      0.62      0.70       429
#          17       0.58      0.40      0.47       101
#          18       0.72      0.70      0.71       154
#          19       0.55      0.53      0.54        80
#
#   micro avg       0.73      0.73      0.73      3978
#   macro avg       0.67      0.59      0.62      3978
#weighted avg       0.73      0.73      0.72      3978
#
#Accuracy: 0.7320261437908496
#Cohen's Kappa: 0.6983910459321659
#Took: 835.171005 seconds

# GRUCell: one layer (2st run)
#              precision    recall  f1-score   support
#
#           0       0.55      0.43      0.48        63
#           1       0.60      0.35      0.44        79
#           2       0.69      0.69      0.69       126
#           3       0.75      0.80      0.78       276
#           4       0.52      0.41      0.46        70
#           5       0.49      0.62      0.55       250
#           6       0.78      0.65      0.71       119
#           7       0.86      0.90      0.88       293
#           8       0.54      0.44      0.48        62
#           9       0.74      0.89      0.81       800
#          10       0.93      0.55      0.69        67
#          11       0.82      0.64      0.72       138
#          12       0.75      0.61      0.67        64
#          13       0.93      0.90      0.91       676
#          14       0.73      0.67      0.70        81
#          15       0.63      0.24      0.35        50
#          16       0.77      0.70      0.73       429
#          17       0.50      0.41      0.45       101
#          18       0.69      0.79      0.74       154
#          19       0.65      0.49      0.56        80
#
#   micro avg       0.75      0.75      0.75      3978
#   macro avg       0.69      0.61      0.64      3978
#weighted avg       0.75      0.75      0.74      3978
#
#Accuracy: 0.7468577174459528
#Cohen's Kappa: 0.7154447213783461
#Took: 611.350802 seconds

# -------
# GRUCell: three layers without dropout
#              precision    recall  f1-score   support
#
#           0       0.51      0.41      0.46        63
#           1       0.41      0.33      0.36        79
#           2       0.74      0.67      0.71       126
#           3       0.76      0.78      0.77       276
#           4       0.47      0.49      0.48        70
#           5       0.47      0.70      0.56       250
#           6       0.68      0.58      0.62       119
#           7       0.83      0.87      0.85       293
#           8       0.52      0.48      0.50        62
#           9       0.78      0.85      0.81       800
#          10       0.84      0.55      0.67        67
#          11       0.73      0.72      0.73       138
#          12       0.72      0.75      0.73        64
#          13       0.95      0.88      0.91       676
#          14       0.45      0.84      0.59        81
#          15       0.39      0.32      0.35        50
#          16       0.88      0.61      0.72       429
#          17       0.54      0.43      0.48       101
#          18       0.73      0.67      0.70       154
#          19       0.56      0.61      0.58        80
#
#   micro avg       0.73      0.73      0.73      3978
#   macro avg       0.65      0.63      0.63      3978
#weighted avg       0.75      0.73      0.73      3978
#
#Accuracy: 0.7332830568124685
#Cohen's Kappa: 0.7024878639964016
#Took: 1888.460148 seconds

# -------
# GRUCell: one layer with SMOTE
#              precision    recall  f1-score   support
#
#           0       0.50      0.08      0.14        63
#           1       0.20      0.06      0.10        79
#           2       0.91      0.08      0.15       126
#           3       0.00      0.00      0.00       276
#           4       0.00      0.00      0.00        70
#           5       0.32      0.23      0.27       250
#           6       0.65      0.14      0.23       119
#           7       0.96      0.08      0.14       293
#           8       1.00      0.05      0.09        62
#           9       0.79      0.30      0.44       800
#          10       0.80      0.06      0.11        67
#          11       0.82      0.07      0.12       138
#          12       1.00      0.06      0.12        64
#          13       0.95      0.20      0.33       676
#          14       0.57      0.15      0.24        81
#          15       0.10      0.04      0.06        50
#          16       0.58      0.14      0.22       429
#          17       0.14      0.07      0.09       101
#          18       0.00      0.00      0.00       154
#          19       0.03      0.99      0.05        80
#
#   micro avg       0.17      0.17      0.17      3978
#   macro avg       0.52      0.14      0.14      3978
#weighted avg       0.62      0.17      0.23      3978
#
#Accuracy: 0.16968325791855204
#Cohen's Kappa: 0.1313228144901376
#Took: 2644.856172 seconds
