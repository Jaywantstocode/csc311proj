import sys
sys.path.append("..")
from utils import *
import numpy as np
from sklearn.impute import SimpleImputer 
import matplotlib as plt
from item_response import sigmoid, update_theta_beta, neg_log_likelihood, irt


def bootstrapping(data):
    """
    Generate a bagged dataset by randomly sampling with replacement
    """
    n = len(data['user_id'])
    index = np.random.randint(0, n, n)
    new_data = {}
    for key in data:
        new_data[key] = np.array(data[key])[index]
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
    np.random.seed(69)

    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    
    parameters = {
        "iterations" : 10,
        "learing rate": 0.01,
        "num_models": 3
    }

    models_list = []

    for i in range(parameters["num_models"]):
        print(f"Model: {i + 1}")
        bagged = bootstrapping(train_data)
        models_list.append(irt(sparse_matrix, bagged, val_data, \
                    parameters["learing rate"], parameters["iterations"])[0:2])
    
    # Get train data bagged predictions and get training accuracy
    train_bagged_predictions = bag_predict(train_data, models_list)
    bagged_train_accuracy = evaluate(train_bagged_predictions, train_data['is_correct'])

    # Get validation data bagged predictions and get validation accuracy
    valid_bagged_predictions = bag_predict(val_data, models_list)
    bagged_valid_accuracy = evaluate(valid_bagged_predictions, val_data['is_correct'])

    # Get train data bagged predictions and get training accuracy
    test_bagged_predictions = bag_predict(test_data, models_list)
    bagged_test_accuracy = evaluate(test_bagged_predictions, test_data['is_correct'])


    print(f"Final Training Accuracy: {bagged_train_accuracy}")
    print(f"Final Validation Accuracy: {bagged_valid_accuracy}")
    print(f"Final Testing Accuracy: {bagged_test_accuracy}")

if __name__ == "__main__":
    main()