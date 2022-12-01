<<<<<<< HEAD
from utils import *
import numpy as np
from sklearn.impute import SimpleImputer 
import matplotlib as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))
=======
# TODO: complete this file.
import sys
sys.path.append("..")

from utils import load_train_csv, load_public_test_csv, load_valid_csv
import numpy as np

parameters = {
    "iterations" : 10,
    "learing rate": 0.01,
}


def bag():
    pass

def bootstrapping():
    pass


def evaluate():
    pass

def bag_predict():
    pass


def predict():
    pass


def main():
    
    pass
>>>>>>> 9ab06f9a539aa02ec8db9bba20844e356bc0a6ad
