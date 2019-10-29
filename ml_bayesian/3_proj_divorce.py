import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat
import pandas as pd
import csv

def sigmoid(a):
    return (1 / (1 + np.exp(-a)))

def getWeightsIRLS(w_old, phi, targets):
    N = targets.shape[0]
    print(phi.shape)

    # Construct the iota matrix (N x M) with rows of phi.T
    iota = phi.T
    print(iota.shape)

    # Initialize parameters that will be updated in IRLS loop
    R = np.eye(N)
    print(R.shape)
    Y = np.zeros((N, 1))
    # difference = 1

    # print(targets[0].shape)

    # IRLS Algorithm - complete when the difference between the data points is very small
    for i in range(1000):
    # while difference > 0.0001:
        counter = -1

        # Construct Y matrix (N x 1) and R matrix (N x N), Eq. 4.98
        for target in np.nditer(targets):
            counter += 1
            # y = sigmoid(np.matmul(w_old.T, phi[:,counter])) # Eq. 4.91
            y = np.matmul(w_old.T, phi[:,counter]) # Eq. 4.91
            # print(y)
            R[counter,counter] = y
            Y[counter,0] = y

        # Eq. 4.100 to construct Z matrix
        z = np.matmul(iota, w_old) - np.matmul(np.linalg.inv(R), (Y - targets))

        # Eq. 4.99 for Newton Raphson Gradient Descent
        w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
        w_new = np.matmul(w_new, iota.T)
        w_new = np.matmul(w_new, R)
        w_new = np.matmul(w_new, z)
        # difference = np.abs(w_old[1,0] - w_new[1,0])
        w_old = w_new
        # print(w_new)
        # print(difference)

    return w_new

def getWeights(data, targets, p_C1):
    N = targets.shape[0]

    # Find ML mean for each class
    N1 = 0
    N2 = 0
    mu1 = 0
    mu2 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        if target == 1: # Class 1
            N1 += 1
            mu1 +=  targets[:,counter] * data[:,counter] # Eq. 4.75
            # print("target was 1: N1 = {}, mu1 = {}, counter is {}".format(N1,mu1,counter))
        else: # Class 2
            N2 += 1
            mu2 +=  (1 - targets[:,counter]) * data[:,counter] # Eq. 4.76
            # print("target was 0: N2 = {}, mu2 = {}, counter is {}".format(N2,mu2,counter))
    mu1 = (1/N1)*mu1 # Eq. 4.75
    # print(mu1)
    mu2 = (1/N2)*mu2 # Eq. 4.76
    # print(mu2)

    # Find covariance for classes (use same covariance for each)
    S1 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        if target == 1: # Class 1
            S1 += np.matmul((data[:,counter] - mu1), np.transpose(data[:,counter] - mu1)) # Eq. 4.79
    S1 = (1/N1) * S1 # Eq. 4.79

    S2 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        if target == 0: # Class 2
            S2 += np.matmul((data[:,counter] - mu2), np.transpose(data[:,counter] - mu2)) # Eq. 4.80
    S2 = (1/N2) * S2 # Eq. 4.80
    S = (N1/N)*S1 + (N2/N)*S2 # Eq. 4.78

    # Find the weights
    sigma = S
    # w0 is a prior, can set the initial probabilities how you desire
    # p_C1 = 0.5
    p_C2 = 1 - p_C1
    w0 = -0.5 * np.matmul(np.matmul(mu1.T, np.linalg.inv(sigma)), mu1) + 0.5 * np.matmul(np.matmul(mu2.T, np.linalg.inv(sigma)), mu2) \
        + np.log(p_C1 / p_C2) # Eq. 4.67
    w0 = w0[0,0]
    w = np.matmul(np.linalg.inv(sigma), mu1 - mu2) # Eq. 4.66
    w = np.array([[w0],[w[0,0]],[w[1,0]]]) # store weights into a vector
    return w

if __name__ == "__main__":

    ##########################################################################################
    ################################## Divorce!!! ############################################
    ##########################################################################################

    # Data obtained through UCI, see here https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
    data_path = 'divorce.csv'

    # The data has features of M = 54 with total datapoints N = 170
    divorce_data = np.loadtxt(data_path, delimiter=';', skiprows=1)

    # extract the targets in the last column and convert to dimension N x 1
    divorce_targets = np.array([divorce_data[:,-1]])
    divorce_targets = divorce_targets.T
    print(divorce_targets.T.shape)

    # remove last column from original dataset and reshape into dimension M x N
    divorce_data = divorce_data[:,:-1]
    divorce_data = divorce_data.T
    print(divorce_data.shape)

    N = divorce_targets.shape[0]
    print(N)

    # # Initialize a figure to plot circles data
    # fig = plt.figure(5)
    # plt.xlim(-1,3)
    # plt.ylim(-1,3)
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.title('Classification of Circles Data')
    # counter = -1

    # # Plot points based on the target value
    # for target in np.nditer(divorce_targets):
    #     counter += 1
    #     if target == 1:
    #         plt.scatter(divorce_data[0,counter], divorce_data[1,counter], marker='x', color = 'r') # Class 1, t = 1
    #     else:
    #         plt.scatter(divorce_data[0,counter], divorce_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 0

    # # Find the ML estimate of the weights using a prior based on the probability that a target is in class 1
    # P_C1 = 0.5
    # circles_w = getWeights(divorce_data, divorce_targets, P_C1)

    
    # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    # divorce_w = np.ones((54, 1))
    divorce_w = np.random.rand(54, 1)
    # print(divorce_w)
    # print(divorce_w.shape)
    divorce_w_irls = getWeightsIRLS(divorce_w, divorce_data, divorce_targets)
    print(divorce_w_irls.shape)


    # Classify data with gaussian generative model and report success
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(divorce_targets):
        counter += 1
        p_class = np.matmul(divorce_w_irls.T, divorce_data[:,counter])
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == divorce_targets[counter,0]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('divorce', 'logistic regression', success / (success + failure)))

    # Initialize a figure to plot an ROC curve for circles data
    fig = plt.figure(1)
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot for Divorce Data')

    thresh = np.linspace(0, 1, 100)
    true_pos_rate = []
    false_pos_rate = []
    # threshold = 0.4
    counter = -1
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    for target in np.nditer(divorce_targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1
    print(N)
    print(P)

    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(divorce_targets):
            counter += 1
            p_class = np.matmul(divorce_w_irls.T, divorce_data[:,counter]) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1 
            else:
                target_guess = 0
            if target_guess == divorce_targets[counter,0] and divorce_targets[counter,0] == 1:
                TP += 1
            elif target_guess == 1 and divorce_targets[counter,0] == 0:
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'g')

    plt.show()