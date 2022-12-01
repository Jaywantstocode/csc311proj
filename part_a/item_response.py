from utils import *
import numpy as np
from sklearn.impute import SimpleImputer 
import matplotlib as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    # print(len(data["user_id"]), theta.shape, len(data["question_id"]), beta.shape)
    # assert(len(data["user_id"]) == theta.shape[1])
    # assert(len(data["question_id"]) == beta.shape[1])
    for i in range(theta.shape[0]):
        for j in range(beta.shape[0]):
            if data["is_correct"][j] == 1:
                log_lklihood += (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j]))
            else:
                log_lklihood += np.log(1 - sigmoid(theta[i]-beta[j]))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    new_theta = np.zeros(theta.shape)
    new_beta = np.zeros(beta.shape)
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        new_theta[u] = 1 - sigmoid(theta[u] - beta[q])
        new_beta[q] = sigmoid(theta[u] - beta[q]) - 1
    # for i in range(len(theta)):
    #     for j in range(len(beta)):
    #         new_theta[i] = 1 - sigmoid((theta[i] - beta[j]))
    #         new_beta[j] = sigmoid((theta[i] - beta[j])) - 1
    
    theta -=  lr * new_theta
    beta -= lr * new_beta    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(matrix, data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    # print(data["user_id"][0], data["question_id"][0], data["is_correct"][0])
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
        # print(theta.shape, beta.shape)
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llk.append(neg_lld)

        val_llk_lst.append(neg_log_likelihood(val_data, theta, beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_llk, val_acc_lst, val_llk_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    weight_reg = 0
    iterations = 500
    learning_rate = 0.001

    theta, beta, train_llk, val_acc_lst, val_llk_lst = irt(sparse_matrix, train_data, val_data, learning_rate, iterations) 
    acc = evaluate(val_data, theta, beta)
    print(f"The validation accuracy is {acc}")
    iteration = [i for i in range(1, len(iterations) + 1)]
    plt.plot(iteration, val_llk_lst, marker = 'o', label='validation llk')
    plt.plot(iteration, train_llk, marker = 'o', label='training llk')
    plt.legend(loc = 'upper right')
    plt.xlabel("iterations")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood for training and validation set")
    plt.show()
    plt.savefig("parta_q2_validation (b).png")

    thetaT, betaT, test_acc_lst = irt(sparse_matrix, train_data, test_data, learning_rate, iterations) 
    accT = evaluate(test_data, thetaT, betaT)
    print(f"The testing accuracy is {accT}")
    pass


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    theta_lst = [theta[3], theta[17], theta[30]]
    plt.plot(theta_lst, sigmoid(theta_lst, beta[3]), color = 'blue', label = 'j1')
    plt.plot(theta_lst, sigmoid(theta_lst, beta[17]), color = 'orange', label = 'j2')
    plt.plot(theta_lst, sigmoid(theta_lst, beta[30]), color = 'red', label = 'j3')

    plt.xlabel('Theta')
    plt.ylabel('Probability of the correct response')
    plt.legend(loc = 'upper right')
    plt.title('Probability of correctly answering 3 Questions Given Student Ability Theta')

    plt.savefig('plt.savefig("parta_q2 (d).png")')
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
