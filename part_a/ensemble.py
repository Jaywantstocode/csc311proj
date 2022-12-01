import sys
sys.path.append("..")
from utils import *
import numpy as np
from sklearn.impute import SimpleImputer 
import matplotlib as plt
from item_response import sigmoid, update_theta_beta, neg_log_likelihood, irt

parameters = {
    "iterations" : 10,
    "learing rate": 0.01,
}


def bootstrapping(matrix):
    """
    Generate a bagged dataset by randomly sampling with replacement
    """
    n = len(matrix['user_id'])
    index = np.random.randint(0, n, n)
    new_data = {}
    for keys in matrix.keys():
        new_data[keys] = np.array(matrix[keys])[index]
    return new_data
    
def predict(data, theta, beta):
    """ Evaluate the model given data and return prediction probabilities
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: Vector (float)
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred

def bag_predict(data, models):
    """
    Get an average accuracy over each of the models.
    """
    preds = np.zeros(len(data['is_correct']))
    for model in models: 
        preds += predict(data, model[0], model[1])
    return preds/len(models)


def evaluate(preds, targets):
    # NEEDA change this into just calculating the accuracy for from bagged_predict
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    return np.sum((np.array(preds >= 0.5) == np.array(targets))) \
           / len(preds)
    


def main():
    
    pass
