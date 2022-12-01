from utils import *
import numpy as np
from sklearn.impute import SimpleImputer 
import matplotlib as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))