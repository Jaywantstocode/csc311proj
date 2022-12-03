from utils import *

import numpy as np
import matplotlib.pyplot as plt

import random
import ast
import pandas as pd
from original_item_response import sigmoid

# size of students, questions and subjects
N, D, S = 542, 1774, 388

def neg_log_likelihood(data, theta, beta, c, k ):
    """ Compute the negative log-likelihood.
    You may optionally replace the function arguments to receive a matrix.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param c: float
    :param k: Vector
    :return: float
    """
    log_lklihood = 0

    # # N x D matrix: for each student, the avg familiarity on the subjects included by each question.
    # alpha_mat = (alpha @ q_meta.T) / np.count_nonzero(q_meta, axis=1)

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]

        # prob = p(c_uq|theta_u, beta_q)
        x = k[q] * (theta[u] - beta[q] ) #+ alpha_mat[u][q])
        prob = sigmoid(x) * (1-c) + c

        correct = data["is_correct"][i]
        log_lklihood += correct * np.log(prob) + (1-correct) * np.log((1-prob))
    return -log_lklihood


def update_parameters(train_matrix, zero_train_matrix, lr, theta, beta, alpha, c, k, lambd, q_meta):
    """ Update theta, beta alpha, c and k using gradient descent.
    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta
    :param train_matrix: N x D matrix
    :param zero_train_matrix: N x D matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param c: float
    :param k: Vector
    :param lambd: float
    :return: tuple of vectors
    """
    # create matrix representation for calculation convenience
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))
    beta_mat = np.ones((N, 1)) @ np.expand_dims(beta, axis=0)
    k_mat = np.ones((N, 1)) @ np.expand_dims(k, axis=0)
    # N x D matrix: for each student, the avg familiarity on the subjects included by each question.
    # alpha_mat = (alpha @ q_meta.T) / np.count_nonzero(q_meta, axis=1)

    nan_mask = np.isnan(train_matrix)

    # update theta
    x = (theta_mat - beta_mat) * k_mat
    # probabilities p(c_uq|theta_u, beta_q) for each u and q
    prob = sigmoid(x) * (1-c) + c
    prob[nan_mask] = 0
    # d_theta = sum_q_[prob-correct] * (1-c) + lambd * theta
    theta -= lr * (np.sum((prob - zero_train_matrix) * k_mat, axis=1) * (1-c) + lambd * theta)
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))

    # update beta
    # d_beta = sum_u_[prob-correct] * (1-c) + lambd * beta
    beta -= lr * (np.sum((zero_train_matrix - prob) * k_mat, axis=0) * (1-c) + lambd * beta)
    beta_mat = np.ones((N, 1)) @ np.expand_dims(beta, axis=0)



    # update k
    # d_k = sum_u_[(prob-correct)*(theta - beta + alpha)] * (1-c) + lambd * k
    k -= lr * (np.sum((prob - zero_train_matrix) * (theta_mat - beta_mat), axis=0) * (1-c) + lambd * k) 
    k_mat = np.ones((N, 1)) @ np.expand_dims(k, axis=0)

    # update c
    c -= lr * (1 - sigmoid(x).mean())
    # constrain c in range [0, 1]
    c = max(0, c)
    c = min(1, c)

    return theta, beta, alpha, c, k


def irt(data, train_matrix, zero_train_matrix, val_data, lr, iterations, lambd):
    """ Train IRT model.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param train_matrix: N x D matrix
    :param zero_train_matrix: N x D matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param lambd: float
    :return: (theta, beta, alpha, c, k)
    """
    theta = np.ones(N) * 0.5
    beta = np.ones(D) * 0.5
    c = 0.25
    k = np.ones((D,)) * 0.5

    # training and validation negtive log likelihoods
    train_neg_llds, val_neg_llds = [], []
    # training and validation accuracies
    train_accs, val_accs = [], []

    for i in range(iterations):

        train_neg_llds.append(neg_log_likelihood(data, theta, beta, c, k))
        val_neg_llds.append(neg_log_likelihood(val_data, theta, beta, c, k))

        theta, beta, c, k = update_parameters(train_matrix, zero_train_matrix, lr, theta, beta, c, k, lambd)
        
        val_acc = evaluate(val_data, theta, beta, c, k)
        train_acc = evaluate(data, theta, beta, c, k)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print("Epoch: {}/{} \t "
              "Train Acc: {}\t"
              "Valid Acc: {}".format(i, iterations, train_acc, val_acc))

    # plot training and validation negtive log likelihoods for each iteration
    plt.figure(1)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), train_neg_llds, 'r')
    ax.plot(np.arange(iterations), val_neg_llds, 'g')
    ax.xaxis.set_label_text('iterations')
    ax.yaxis.set_label_text('negtive log likelihoods')
    ax.set_title('irt_update_a: train and val negtive log likelihoods for each iteration')
    ax.legend(['train_neg_llds', 'val_neg_llds'])
    plt.savefig('./part b. irt_update_a: train and val negtive log likelihoods for each iteration.png')

    # plot train and val accuracies for each iteration
    plt.figure(2)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), train_accs, 'r')
    ax.plot(np.arange(iterations), val_accs, 'g')
    ax.xaxis.set_label_text('iteration')
    ax.yaxis.set_label_text('accuracies')
    ax.set_title('irt_update_a: train and val accuracies for each iteration')
    ax.legend(['train_accs', 'val_accs'])
    plt.savefig('./part b. irt_update_a: train and val accuracies for each iteration.png')

    return theta, beta, c, k


def evaluate(data, theta, beta, c, k):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param c: float
    :param k: Vector
    :return: float
    """
    pred = []

    # N x D matrix: for each student, the avg familiarity on the subjects included by each question.
    # alpha_mat = (alpha @ q_meta.T) / np.count_nonzero(q_meta, axis=1)

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((theta[u] - beta[q]) * k[q]).sum()
        p_a = sigmoid(x) * (1-c) + c
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    
    train_matrix = load_train_sparse("../data").toarray()
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0

    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    private_data = load_private_test_csv("../data")

    lr = 0.02
    lambd = 0.1
    iterations = 60
    theta, beta, alpha, c, k = irt(train_data, train_matrix, zero_train_matrix, val_data, lr, iterations, alpha, lambd)

    # part (c) report the fianl validation and test accuracies
    final_val_acc = evaluate(val_data, theta, beta, alpha, c, k)
    final_test_acc = evaluate(test_data, theta, beta, alpha, c, k)
    print(f"The final validation accuracy is {final_val_acc}, the final test accuracy is {final_test_acc}")

    final_train_acc = evaluate(train_data, theta, beta, alpha, c, k)  
    print(f"The final training accuracy is {final_train_acc}")

    # private test
    pred = []
    for i, q in enumerate(private_data["question_id"]):
        u = private_data["user_id"][i]
        x = ((theta[u] - beta[q]) * k[q]).sum()
        p_a = sigmoid(x) * (1 - c) + c
        pred.append(p_a >= 0.5)
    private_data['is_correct'] = pred
    save_private_test_csv(private_data)


if __name__ == "__main__":
    main()

