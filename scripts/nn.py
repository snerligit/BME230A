import sys
import numpy as np
from matplotlib import pyplot
import pandas as pd
import h5py
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

print("USING pytorch VERSION: ", torch.__version__)

y_index_key = {'native': 0, 'close': 1, 'moderate': 2, 'far': 3}
#y_index_key = {'native': 0, 'non-native': 1}

for name in y_index_key:
  print(name, y_index_key[name])


class PrimarySiteDataset(Dataset):

    '''
    Class inherited from PyTorch Dataset class.

    Purpose: This class provides methods to return data from PyTorch tensor format to numpy format

    Attributes:

    x_data: Training data
    y_data: Testing data
    length: length of training data
    '''

    def __init__(self, x_path, y_path, batch_size=None):

        '''
        constructor for this class

        parameters:
        x_path: path to the training data
        y_path: path to the testing data
        batch_size: used to load data in batches instead of individual samples (This is not used here)

        '''
        x = np.loadtxt(open(x_path, "rb"), delimiter=",", skiprows=0)
        y = np.loadtxt(open(y_path, "rb"), delimiter=",", skiprows=0)

        print (x)

        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor     # for MSE Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

        # Therefore, for my example, I will have:
        # Dims of x: 52900 instances x 3201 features
        # Dims of y: 52900 instances x 4 class labels

    def __getitem__(self, index):

        '''
        overwriting the getitem method provided by Dataset class

        parameters:
        index: index of required data in x and y

        returns:
        the values in x_data and y_data at a particular index

        '''
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        '''
        overwriting the len method provided by Dataset class

        parameters: None

        returns:
        the length of x

        '''
        return self.length

class ShallowLinear(nn.Module):
    '''
    A simple, general purpose, fully connected network which inherits from Module

    Purpose: This class is a shallow network consisting of an input layer, a hidden layer and an output layer.

    Attributes:

    linear1: a linear transformation object to apply tranformation from input layer to hidden layer
    linear2: a linear transformation object to apply tranformation from hidden layer to output layer

    '''
    def __init__(self):
        # Perform initialization of the pytorch superclass
        super(ShallowLinear, self).__init__()

        '''
        constructor for ShallowLinear class

        '''

        # Define network layer dimensions
        #D_in, H1, H2, H3, D_out = [3201, 32, 64, 32, 4]    # These numbers correspond to each layer: [input, hidden_1, output]
        #D_in, H1, H2, H3, D_out = [361, 32, 64, 32, 4]    # These numbers correspond to each layer: [input, hidden_1, output]
        #D_in, H1, H2, H3, D_out = [163, 32, 64, 32, 4]    # These numbers correspond to each layer: [input, hidden_1, output] -hydrophobic peptide only
        #D_in, H1, H2, H3, D_out = [361, 32, 64, 32, 2]    # These numbers correspond to each layer: [input, hidden_1, output]
        D_in, H1, H2, H3, D_out = [1441, 32, 64, 32, 4]    # These numbers correspond to each layer: [input, hidden_1, output] - hydrophobic + charged

        # Define layer types
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        '''
        This method defines the network layering and activation functions

        parameters:
        x: input variable

        returns:
        the transformed x from passing through hidden layer and the output layer

        '''
        x = self.linear1(x) # hidden layer
        x = torch.relu(x)       # activation function

        x = self.linear2(x) # output layer
        x = torch.relu(x)       # activation function

        x = self.linear3(x) # output layer
        x = torch.relu(x)       # activation function

        x = self.linear4(x) # output layer

        return x

def train_batch(model, x, y, optimizer, loss_fn):

    '''
    purpose: This method is used to train a given example x in the train dataset

    parameters:
    model: This is an instance of ShallowLinear class, which defines our neural network architecture
    x: training example
    y: actual class label of the training example, x
    optimizer: an instance of optim class that is used to update the parameters or weights based on gradients
    loss_fn: penalty to apply if the model fails to classify a given instance correctly

    returns:
    loss: the loss value after training example x
    y_predict: predicted y value to compute training distribution/accuracy

    '''
    # Run forward calculation
    y_predict = model.forward(x)

    # convert 1-hot vectors back into indices
    max_values, target_index = y.max(dim = 1)
    target_index = target_index.type(torch.LongTensor)
    #target_index = y.type(torch.LongTensor)

    # Compute loss.
    #loss = loss_fn(y_predict, y)
    loss = loss_fn(y_predict, target_index)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data.item(), y_predict


def train(model, loader, optimizer, loss_fn, epochs=5):

    '''
    purpose: This method is used to train the neural network using all the training dataset. This method in turn calls
    train_batch() for every example in the dataset

    parameters:
    model: This is an instance of ShallowLinear class, which defines our neural network architecture
    loader: instance of DataLoader class in PyTorch. This utility class holds train dataset in its attributes
    optimizer: an instance of optim class that is used to update the parameters or weights based on gradients
    loss_fn: penatly to apply if the model fails to classify a given instance correctly
    epochs: number of times to loop over the dataset

    returns:
    losses: set of loss values for all the training examples

    '''
    losses = list()
    y_argmax = list()
    y_predict_argmax = list()
    first_argmax_list = True

    first_predict = list()
    first = list()
    first_val = True

    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss, y_predict = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
            losses.append(loss)

            if first_val:
              first_predict = y_predict
              first = y
              first_val = False

            if first_argmax_list:
              for i in range(0, len(y)):
                y_predict_argmax.append(np.argmax(y_predict[i].tolist()))
                y_argmax.append(np.argmax(y[i].tolist()))


            batch_index += 1

        first_argmax_list = False

        print("Epoch: ", e+1)
        print("Batches: ", batch_index)

    '''

    A set of print statements to answer the questions from homework

    print ("Length of training examples:", len(y_argmax), len(y_predict_argmax))
    print ("First training lists: ", first_predict[0], first[0])
    print ("y_predict, y")
    for i in range(0, len(first[0])):
      print (round(first_predict[0].tolist()[i], 3), round(first[0].tolist()[i], 0))
    print ("Shape of argmax vectors: ", len(y_predict_argmax), len(y_argmax))
    print ("First training argmax: ", y_predict_argmax[0], y_argmax[0])
    accuracy = compute_accuracy(y_argmax, y_predict_argmax)
    print ("Accuracy: ", accuracy)
    plot_histogram(y_argmax)
    plot_histogram(y_predict_argmax)

    '''

    return losses


def test_batch(model, x, y):
    '''
    purpose: This method is used to assign class label to each example in the test dataset
    model: This is an instance of ShallowLinear class, which defines our neural network architecture

    parameters:
    x: test data example
    y: original class label of the test data example x

    returns:
    y_predict: predicted class label of the test example input x

    '''
    # run forward calculation
    y_predict = model.forward(x)

    return y_predict

def test(model, loader):

    '''
    purpose: This method is used to run the neural network with 20% of the test data reserved in the previous methods
    model: This is an instance of ShallowLinear class, which defines our neural network architecture
    loader: instance of DataLoader class in PyTorch. This utility class holds test dataset in its attributes

    returns:
    y_vector: Original class labels of the test dataset
    y_predict_vector: predicted class lables of the test dataset

    '''
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y_predict = test_batch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)
    y_vector = np.concatenate(y_vectors)

    return y_vector, y_predict_vector

def metrics(obs, pred):

  '''
  purpose: This method is used to compute the confusion matrix and print classification report (precision, recall and F1 measure) using observed and predicted labels

  parameters:
  obs: observed or original labels
  pred: predicted labels from a neural network

  returns:
  cm: confusion matrix

  '''

  labels = []
  for key in y_index_key:
    labels.append(key)

  cm = confusion_matrix(obs, pred)
  print ("Confusion matrix")
  print (cm)

  print ("Heatmap showing the confusion matrix")
  fig, ax = pyplot.subplots()
  im = ax.imshow(cm)

  # We want to show all ticks...
  ax.set_xticks(np.arange(len(cm)))
  ax.set_yticks(np.arange(len(cm)))
  # ... and label them with the respective list entries
  ax.set_xticklabels(labels, fontsize=6)
  ax.set_yticklabels(labels, fontsize=6)

  # Rotate the tick labels and set their alignment.
  pyplot.setp(ax.get_xticklabels(), rotation=90, ha="right",
          rotation_mode="anchor")

  fig.colorbar(im, ax=ax)

  pyplot.show()

  #In addition, show the classification report for each class which gives precision, recall and f1-score
  cr = classification_report(obs, pred, target_names=labels)
  print ("Classification report for each class")
  print (cr)


  return cm


def compute_accuracy(y, y_predict):

  '''
  purpose: This method is used to report the accuracy of the model using original and predicted values of y.
  It is redundant since classification report in metrics() function already report accuracy.

  parameters:
  y: original class labels
  y_predict: predicted class labels

  returns:
  accuracy: accuracy of classification calculated as the ratio of correctly identified labels to all the identified labels

  '''
  match = 0
  mismatch = 0
  for i in range(0, len(y)):
    if y[i] == y_predict[i]:
      match += 1
    else:
      mismatch += 1

  accuracy = match/(match+mismatch)
  return accuracy

def scatter_plot(x, y):

  '''
  purpose: This method is used to draw the scatter plot showing the relation between x and y variables

  Added this method just to answer part 5, question 5 of the homework

  parameters:
  x: accuracy of the model prediction
  y: frequency of observed or predicted labels for the test dataset

  returns: Nothing

  '''

  fig = pyplot.figure()
  fig.set_size_inches(8,6)
  ax = pyplot.axes()
  ax.set_xlabel("Accuracy of class prediction")
  ax.set_ylabel("Frequency of the class in the test dataset")
  pyplot.scatter(x, y, marker='o', s=20)
  pyplot.show()

def frequency(y):

  '''
  purpose: This method is used to compute the frequencies of all the classes in a dataset.

  Added this method just to answer part 5, question 5 of the homework

  parameters:
  y: observed or predicted labels for the test dataset

  returns:
  freq_arr: array containing the frequencies of each class

  '''

  freq_table = {}

  for i in range(0, len(y)):
    if y[i] in freq_table:
      freq_table[y[i]] += 1
    else:
      freq_table[y[i]] = 1

  freq_arr = np.zeros(len(freq_table))
  for key in freq_table:
    freq_arr[key] = freq_table[key]

  return freq_arr

def run(dataset_train, dataset_test):

    '''
    purpose: This method takes as input the train and test datasets, defines the hyperparameters,
    calls train and test methods to train and test the neural network.

    parameters:
    dataset_train: training dataset
    dataset_test: testing dataset

    returns:
    y_vec: The original labels of the testing dataset
    y_predict: The predicted labels of the testing dataset
    loss: cross entropy loss for each iteration of the training dataset

    '''
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 32

    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=False)

    # Define the hyperparameters
    learning_rate = 1e-4
    shallow_model = ShallowLinear()

    n_epochs = 4

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(shallow_model.parameters(), lr=learning_rate)

    # Define the loss function
    #loss_fn = nn.MSELoss()  # mean squared error
    loss_fn = nn.CrossEntropyLoss() # cross entropy loss

    # Train and get the resulting loss per iteration
    loss = train(model=shallow_model, loader=data_loader_train, optimizer=optimizer, loss_fn=loss_fn, epochs=n_epochs)

    # Test and get the resulting predicted y values
    y_vec, y_predict = test(model=shallow_model, loader=data_loader_test)

    print (y_predict)

    return loss, y_vec, y_predict

def main():

  '''
  The following block of code defines the datasets for training and testing and calls run() method.

  '''

  #x_train_path = "train/groove_pep_union/x_train.csv"
  #y_train_path = "train/groove_pep_union/y_train.csv"
  #x_test_path = "test/groove_pep_union/x_test.csv"
  #y_test_path = "test/groove_pep_union/y_test.csv"
  #x_train_path = "train/pep_only/x_train.csv"
  #y_train_path = "train/pep_only/y_train.csv"
  #x_test_path = "test/pep_only/x_test.csv"
  #y_test_path = "test/pep_only/y_test.csv"
  #x_train_path = "train/hydrophobic/x_train.csv"
  #y_train_path = "train/hydrophobic/y_train.csv"
  #x_test_path = "train/hydrophobic/x_train.csv"
  #y_test_path = "train/hydrophobic/y_train.csv"
  #x_train_path = "dihedrals/x_train.csv"
  #y_train_path = "dihedrals/y_train.csv"
  #x_test_path = "dihedrals/x_train.csv"
  #y_test_path = "dihedrals/y_train.csv"

  x_train_path = "x_train.csv"
  y_train_path = "y_train.csv"
  x_test_path = "x_train.csv"
  y_test_path = "y_train.csv"
  #x_test_path = "x_test.csv"
  #y_test_path = "y_test.csv"
  dataset_train = PrimarySiteDataset(x_path=x_train_path,y_path=y_train_path)
  dataset_test = PrimarySiteDataset(x_path=x_test_path,y_path=y_test_path)

  print("Train set size: ", dataset_train.length)
  print("Train set features: ", len(dataset_train[0]))
  print("Test set size: ", dataset_test.length)

  losses, y_vec, y_predict = run(dataset_train=dataset_train, dataset_test=dataset_test)
  y_vec_argmax = list()
  y_predict_argmax = list()

  for i in range(0, len(y_vec)):
    y_vec_argmax.append(np.argmax(y_vec[i]))
    y_predict_argmax.append(np.argmax(y_predict[i]))

  freq_table = frequency(y_vec_argmax)
  cm = metrics(y_vec_argmax, y_predict_argmax)
  accuracy = compute_accuracy(y_vec_argmax, y_predict_argmax)

  class_accuracy = []
  for i in range(0, len(cm)):
    class_accuracy.append(cm[i][i])

  #print ("Scatter plot showing the correlation between frequency of the class and the model performance for that class")
  #scatter_plot(class_accuracy, freq_table)

  print ("Accuracy of the test dataset: ", accuracy)

if __name__ == "__main__":

  main()
