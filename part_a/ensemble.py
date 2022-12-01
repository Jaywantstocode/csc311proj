import sys
sys.path.append("..")
from utils import *
import numpy as np
from sklearn.impute import SimpleImputer 
import matplotlib as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


parameters = {
    "iterations" : 10,
    "learing rate": 0.01,
}

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.

    for i in range(theta.shape[0]):
        for j in range(beta.shape[0]):
            if data["is_correct"][j] == 1:
                log_lklihood += (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j]))
            else:
                log_lklihood += np.log(1 - sigmoid(theta[i]-beta[j]))
    return -log_lklihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    new_theta = np.zeros(theta.shape)
    new_beta = np.zeros(beta.shape)
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        new_theta[u] = 1 - sigmoid(theta[u] - beta[q])
        new_beta[q] = sigmoid(theta[u] - beta[q]) - 1
    
    theta -=  lr * new_theta
    beta -= lr * new_beta    
    return theta, beta

def irt(matrix, data, val_data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    format = SimpleImputer()
    format.fit(matrix)
    question = format.transform(matrix)
    format.fit(matrix.T)
    student = format.transform(matrix.T)

    theta = np.mean(student, axis=0).T
    beta = 1 - np.mean(question, axis=0).T
    
    train_llk = []
    val_acc_lst = []
    val_llk_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llk.append(neg_lld)

        val_llk_lst.append(neg_log_likelihood(val_data, theta, beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_llk, val_acc_lst, val_llk_lst


def bootstrapping(matrix):
    """
    Generate a bagged dataset by randomly sampling with replacement
    """
    
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
