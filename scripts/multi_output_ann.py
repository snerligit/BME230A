# mlp for multi-output regression
from numpy import mean
import torch
from numpy import std
from numpy import asarray
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import argparse
from collections import defaultdict

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

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

        #print (x)

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

# get the dataset
def get_dataset():
    #X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=3, random_state=2)

    x_train_path = "dihedrals/a0201/x_train.csv"
    y_train_path = "dihedrals/a0201/y_train.csv"
    #x_test_path = "dihedrals/a0201/x_test.csv"
    #y_test_path = "dihedrals/a0201/y_test.csv"
    x_test_path = "dihedrals/a0201/x_train.csv"
    y_test_path = "dihedrals/a0201/y_train.csv"

    #x_train_path = "dihedrals/x_groove_train.csv"
    #y_train_path = "dihedrals/y_groove_train.csv"
    #x_test_path = "dihedrals/x_groove_test.csv"
    #y_test_path = "dihedrals/y_groove_test.csv"
    #x_test_path = "dihedrals/blind_x_test.csv"
    #y_test_path = "dihedrals/blind_y_test.csv"



    dataset_train = PrimarySiteDataset(x_path=x_train_path,y_path=y_train_path)
    dataset_test = PrimarySiteDataset(x_path=x_test_path,y_path=y_test_path)
    x_train = dataset_train.x_data
    y_train = dataset_train.y_data

    x_test = dataset_test.x_data
    y_test = dataset_test.y_data

    #print (dataset_train)
    return x_train, y_train, x_test, y_test

# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    #model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu',  kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')
    return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=100)
		# evaluate model on test set
		mae = model.evaluate(X_test, y_test, verbose=0)
		# store result
		print('>%.3f' % mae)
		results.append(mae)
	return results

def evaluate_test_set(y_test, y_test_predict, cutoff):
    #print ("Lengths: ", len(y_test), len(y_test_predict))

    for i in range(0, len(y_test)):
        match = 0
        for j in range(0, len(y_test[0])):
            if abs(y_test[i][j] - y_test_predict[i][j]) <= cutoff:
                match += 1
        #print (len(y_test[0]), match)


def identify_templates_closer_to_test(train_pdbids, test_pdbids, y_test_predict, y_train, cutoff, match_cutoff):

    templates_for_test = {}

    for i in range(0, len(y_test_predict)):
        for j in range(0, len(y_train)):
            match = 0
            for k in range(2, len(y_test_predict[0])-2):
                if abs(y_test_predict[i][k] - y_train[j][k]) <= cutoff:
                    match += 1

            #print (len(y_test[0])-4, match)
            if match >= match_cutoff:
                if test_pdbids[i] not in templates_for_test:
                    templates_for_test[test_pdbids[i]] = [train_pdbids[j]]
                else:
                    templates_for_test[test_pdbids[i]] += [train_pdbids[j]]


    for k in templates_for_test:
        #print (k, len(templates_for_test[k]), templates_for_test[k], "\n\n")
        print (k, len(templates_for_test[k]))

    return templates_for_test


def read_pdbid_file(filename):

    pdbids = []
    inputfilehandler = open(filename, 'r')
    for line in inputfilehandler:
        line = line.rstrip()
        pdbids.append(line)
    inputfilehandler.close()
    return pdbids

def read_rmsd_file(filename):

    rmsds = defaultdict(dict)
    peptides = {}
    inputfilehandler = open(filename, 'r')
    header  =inputfilehandler.readline()
    for line in inputfilehandler:
        line = line.rstrip()
        fields = line.split(',')
        pdbid1 = fields[0].split('_')[0]
        pdbid2 = fields[1].split('_')[0]
        peptides[pdbid1] = fields[0].split('_')[1]
        peptides[pdbid2] = fields[1].split('_')[1]
        rmsd = round(float(fields[2]), 3)
        rmsds[pdbid1][pdbid2] = rmsd

    inputfilehandler.close()
    return rmsds, peptides

def homologs(seq1, seq2, aadiff):

    diff = 0
    for i in range(0, len(seq1)):
        if seq1[i] != seq2[i]:
            diff += 1

    if diff >= aadiff:
        return False
    return True

def compare_RMSD_values(rmsds, peptides, templates_for_test, rmsd_cutoff, aadiff):

    match = 0
    matches = {}
    for pdbid1 in templates_for_test:
        for pdbid2 in templates_for_test[pdbid1]:
            if not homologs(peptides[pdbid1], peptides[pdbid2], aadiff) and (rmsds[pdbid1][pdbid2] < rmsd_cutoff):
                matches[pdbid1] = pdbid2
                match += 1
                break

    print ("Total PDBs containing decent templates: ", len(templates_for_test), match)
    #for m in matches:
    #    print (m, matches[m])
    for t in templates_for_test:
        if t not in matches:
            print (t)


def parse_args():

    parser = argparse.ArgumentParser(usage="Please see the options below")
    parser.add_argument("-train_pdbids", help="pdbids in a train text file")
    parser.add_argument("-test_pdbids", help="pdbids in a test text file")
    parser.add_argument("-cutoff", help="cutoff values for angles", type=float, default=40.0)
    parser.add_argument("-match_cutoff", help="cut off values for number of matches", type=int, default=8)
    parser.add_argument("-rmsd", help="rmsd between different templates")
    parser.add_argument("-rmsd_cutoff", help="rmsd cutoff to select optimal templates", type=float, default=1.0)
    parser.add_argument("-eliminate_homologs", default=True, help="remove homologous peptides from the list for evaluation")
    parser.add_argument("-aadiff", default=3, type=int, help="remove homologous peptides from the list for evaluation")

    return parser.parse_args()

# load dataset
x_train, y_train, x_test, y_test = get_dataset()
# evaluate model
results = evaluate_model(x_train, y_train)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))


n_inputs, n_outputs = x_train.shape[1], y_train.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(x_train, y_train, verbose=0, epochs=100)
# make a prediction for new data
y_test_predict = model.predict(x_test)
evaluate_test_set(y_test, y_test_predict, 60)

args = parse_args()
rmsds, peptides = read_rmsd_file(args.rmsd)
train_pdbids = read_pdbid_file(args.train_pdbids)
test_pdbids = read_pdbid_file(args.test_pdbids)
templates_for_test = identify_templates_closer_to_test(train_pdbids, test_pdbids, y_test_predict, y_train, args.cutoff, args.match_cutoff)
compare_RMSD_values(rmsds, peptides, templates_for_test, args.rmsd_cutoff, args.aadiff)
