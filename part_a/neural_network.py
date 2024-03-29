import sys
sys.path.append("..")

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, train_data, valid_data, test_data

def sigmoid(x):
        """ Apply sigmoid function.
        """
        return np.exp(x) / (1 + np.exp(x))

class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    

    
    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        # For the regularizer
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = inputs

        # Encode inputs: 
        code = torch.sigmoid(self.g(inputs))

        # Decode the encoded input
        out = torch.sigmoid(self.h(code))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, train_dic):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    train_loss_lst = []
    val_loss_lst = []
    # Define optimizers and loss function. Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()
            
            # Set the gradients to 0 to prevent accumulating the gradients from multiple passes
            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb/2 * model.get_weight_norm())
            # Computes the dloss/dx 
            loss.backward()

            train_loss += loss.item()
            # Updates the value of optimizer with the gradient 
            optimizer.step()
        
        train_loss_lst.append(eval_loss(model, zero_train_data, train_dic))
        val_loss_lst.append(eval_loss(model, zero_train_data, valid_data))

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return train_loss_lst, val_loss_lst
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        # If the prob from output is >= 0.5, then it is correct. 
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def eval_loss(model, train_data, valid_data):
    """
    Evaluate the valid_data on the current model and get the validation loss
    """
    model.eval()

    total = 0 
    
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        target = valid_data['is_correct'][i]
        guess = output[0][valid_data["question_id"][i]].item()
        total += (guess - target) ** 2.
    # print(total)
    return total / len(valid_data['user_id'])


def main():
    zero_train_matrix, train_matrix, train_data, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # This part is to find what the optimal k and lambdas are
    # k = [10, 50, 100, 200, 500]

    # # Set optimization hyperparameters.
    # lr = 0.01
    # num_epoch = 25
    # lamb = [0.001, 0.01, 0.1, 1]

    # for choice in k:
        # print(train_matrix.shape[1])
        # model = AutoEncoder(train_matrix.shape[1], choice)
        
        # for lam in lamb: 
        #     train(model, lr, lam, train_matrix, zero_train_matrix,
        #         valid_data, num_epoch)
        #     acc = evaluate(model, zero_train_matrix, test_data)
        #     print(f"with lam: {lam} and k: {choice}, accuracy is: {acc}")

        # print(f"with k: {choice}, accuracy is: {acc}")
            

    optimal_k = 200
    lr = 0.01
    epoch = 25
    optimal_lamb = 0.01
    
    model = AutoEncoder(train_matrix.shape[1], optimal_k)
    train_loss, val_loss = train(model, lr, optimal_lamb, train_matrix, zero_train_matrix, valid_data, epoch, train_data)
    val_acc = evaluate(model, zero_train_matrix, valid_data)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Validation accuracy with optimal k: {optimal_k} and optimal lambda: {optimal_lamb} is: {val_acc}")
    print(f"Test accuracy with optimal k: {optimal_k} and optimal lambda: {optimal_lamb} is: {test_acc}")
    plt.title('Training Loss and Validation Loss over Epochs')
    plt.plot(train_loss, color = 'orange', label = 'Training Loss')
    plt.plot(val_loss, color = 'blue', label = 'Validation Loss')
    plt.legend(loc='best')
    plt.ylabel('Mean Squared Error (Loss)')
    plt.xlabel('Epochs')
    plt.savefig("parta_q3(d).png")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
