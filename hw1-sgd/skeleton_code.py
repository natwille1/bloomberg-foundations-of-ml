import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from sklearn.linear_model import LinearRegression

### Assignment Owner: Tian Wang

#######################################
#### Normalization


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    # remove columns with constant values
    remindx = np.where(np.max(train,axis=0)==np.min(train,axis=0))
    trainrel = np.delete(train,remindx,axis=1)
    min, max = np.min(trainrel, axis=0), np.max(trainrel, axis=0)
    # rescale the datasets
    train_norm = (trainrel-min)/(max-min)
    test_norm = (test-min)/(max-min)
    return train_norm, test_norm


########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    #TODO
    preds = X@theta
    loss = np.sum((preds-y)**2)/X.shape[0]
    return loss


########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    preds = X@theta
    loss = preds - y
    # average gradient across all data points
    grad = 2*(X.T@loss)
    avg_grad = grad/X.shape[0]
    return avg_grad


###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO

#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features) #initialize theta
    #TODO
    theta_hist[0,:] = theta
    loss_hist[0] = compute_square_loss(X, y, theta_hist[0, :])
    for i in range(num_iter):
        cur_theta = theta_hist[i,:]
        dtheta = compute_square_loss_gradient(X, y, cur_theta)
        # update weights
        theta_hist[i+1,:] = cur_theta - alpha * dtheta
        # calculate loss with new weights/theta
        loss_hist[i+1] = compute_square_loss(X, y, theta_hist[i+1, :])
    return theta_hist, loss_hist


####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
def backtracking_line_search(X, y, num_iter=1000, b=0.4, a=0.5):
    t=1
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    stepsize_hist = np.zeros(num_iter+1)
    theta = np.zeros(num_features) #initialize theta
    for i in range(num_iter+1):
        loss = compute_square_loss(X, y, theta)
        grad = compute_square_loss_gradient(X, y, theta)
        desc_dir = -grad
        slope = np.dot(grad, desc_dir)
        while True:
            nextloss = compute_square_loss(X, y, theta+t*desc_dir)
            if nextloss <= (loss+t*a*slope):
                theta = theta + t * desc_dir
                theta_hist[i] = theta
                loss_hist[i] = compute_square_loss(X, y, theta)
                stepsize_hist[i] = t
                break
            else:
                t *= b
    return theta_hist, loss_hist, stepsize_hist

###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    # ridge_term = (lambda_reg/X.shape[0])*np.sum(theta@theta)
    grad = compute_square_loss_gradient(X, y, theta) +2*lambda_reg*theta
    return grad

def batch_grad_descent_train(X_train, y_train, X_test, y_test, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X_train.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    train_loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    test_loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    theta_hist[0, :] = theta
    train_loss_hist[0] = compute_square_loss(X_train, y_train, theta)
    test_loss_hist[0] = compute_square_loss(X_test, y_test, theta)
    for i in range(num_iter):
        cur_theta = theta_hist[i, :]
        grad = compute_square_loss_gradient(X_train, y_train,cur_theta)
        theta_hist[i+1,:] = cur_theta - alpha * grad
        train_loss = compute_square_loss(X_train, y_train, theta_hist[i+1,:])
        test_loss = compute_square_loss(X_test, y_test, theta_hist[i+1,:])
        train_loss_hist[i+1] = train_loss
        test_loss_hist[i+1] = test_loss

    return theta_hist, train_loss_hist, test_loss_hist
###################################################
### Batch Gradient Descent with regularization term

def regularized_grad_descent_train(X_train, y_train, X_test, y_test, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X_train.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    train_loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    test_loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    theta_hist[0, :] = theta
    train_loss = compute_square_loss(X_train, y_train, theta)
    test_loss = compute_square_loss(X_test, y_test, theta)
    reg_loss = np.sum(np.square(theta))
    train_loss_hist[0] = train_loss + lambda_reg * reg_loss
    test_loss_hist[0] = test_loss + lambda_reg * reg_loss
    for i in range(num_iter):
        cur_theta = theta_hist[i, :]
        grad = compute_regularized_square_loss_gradient(X_train, y_train,cur_theta, lambda_reg)
        theta_hist[i+1,:] = cur_theta - alpha * grad
        train_loss = compute_square_loss(X_train, y_train, theta_hist[i+1,:])
        test_loss = compute_square_loss(X_test, y_test, theta_hist[i+1,:])
        train_loss_hist[i+1] = train_loss
        test_loss_hist[i+1] = test_loss

    return theta_hist, train_loss_hist, test_loss_hist

def regularized_grad_descent(X_train, y_train, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X_train.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    train_loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    theta_hist[0, :] = theta
    train_loss = compute_square_loss(X_train, y_train, theta)
    reg_loss = np.sum(np.square(theta))
    train_loss_hist[0] = train_loss + lambda_reg * reg_loss
    for i in range(num_iter):
        cur_theta = theta_hist[i, :]
        grad = compute_regularized_square_loss_gradient(X_train, y_train,cur_theta, lambda_reg)
        theta_hist[i+1,:] = cur_theta - alpha * grad
        train_loss = compute_square_loss(X_train, y_train, theta_hist[i+1,:])
        train_loss_hist[i+1] = train_loss

    return theta_hist, train_loss_hist


#############################################
## Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss


#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta


    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    #TODO
    # shuffle X and y
    Xs, ys = shuffle(X, y, random_state=10)
    for i in range(iter):
        for xs,ys in zip(Xs,ys):
            



################################################
### Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value) and/or objective_function_value

def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term

    # TODO
    theta_hist, loss_hist, stepsize_hist = backtracking_line_search(X_train, y_train, num_iter=1000)
    # plt.title("backtracking_line_search")
    plt.plot(loss_hist, label="backtracking")
    # plt.show()
    for i in [0.05, 0.01]:
        theta_hist, loss_hist = batch_grad_descent(X_train, y_train, num_iter=1000, alpha=i)
        plt.plot(loss_hist, label=i)
    plt.legend()
    plt.show()
    theta_hist, train_loss, test_loss = batch_grad_descent_train(X_train, y_train, X_test, y_test, alpha=0.01, num_iter=1000)
    plt.title("no regularization")
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.show()
    theta_hist, train_loss, test_loss = regularized_grad_descent_train(X_train, y_train, X_test, y_test, alpha=0.01, lambda_reg=0.5, num_iter=1000)
    plt.title("regularization")
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.show()
    # fig, ax = plt.subplots(1,2, figsize=(8,3))
    # for l in [10e-7, 10e-5, 10e-3, 10e-1, 1, 10, 100]:
    train_losses, test_losses = [], []
    for l in [0.01, 0.5, 0.1]:
        theta_hist, train_loss_hist = regularized_grad_descent(X_train, y_train, alpha=0.01, lambda_reg=l)
        tlindex = np.argmin(train_loss_hist)
        best_train_loss = train_loss_hist[tlindex]
        best_theta = theta_hist[tlindex]
        test_loss = compute_square_loss(X_test, y_test, best_theta)
        train_losses.append(best_train_loss)
        test_losses.append(test_loss)
    plt.plot([0.01, 0.5, 0.1], train_losses, 'x', label='train')
    plt.plot([0.01, 0.5, 0.1], test_losses, 'x', label='test')
    plt.legend()
    plt.ylabel("squared loss")
    plt.xlabel("lambda value")
    plt.xscale("log")
    plt.show()
    # plt.savefig("imgs/reg_grad_descent_lambda_search.png")
    # plt.close()


    #     ax[0].plot(best_train_loss, 'o',label=l)
    #     ax[0].set_title("train loss")
    #     ax[1].plot(test_loss, 'o', label=l)
    #     ax[1].set_title("test loss")
    # ax[0].legend()
    # ax[1].legend()
    # # ax[0].set_yscale("log")
    # # ax[1].set_yscale("log")
    # ax[0].set_ylabel("squared loss")
    # ax[0].set_xlabel("iteration")
    # plt.show()
    # plt.savefig("imgs/batch_grad_descent_loss.png")
    # plt.close()

if __name__ == "__main__":
    main()
