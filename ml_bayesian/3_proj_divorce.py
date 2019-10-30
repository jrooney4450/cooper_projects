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
    # Format weights into vector
    w1 = w_old[0]
    w2 = w_old[1]
    w_old = np.vstack((w1, w2))

    N = targets.shape[0]

    # Construct the iota matrix (N x M) with rows of phi.T
    iota = phi.T

    # Initialize parameters that will be updated in IRLS loop
    R = np.eye(N)
    Y = np.zeros((N, 1))

    # IRLS Algorithm - complete a set amount of loops
    for i in range(1000):
        counter = -1

        # Construct Y matrix (N x 1) and R matrix (N x N), Eq. 4.98
        for target in np.nditer(targets):
            counter += 1
            y = sigmoid(np.matmul(w_old.T, phi[:,counter])) # Eq. 4.91
            R[counter,counter] = y
            Y[counter,0] = y

        # Eq. 4.100 to construct Z matrix
        z = np.matmul(iota, w_old)
        z = z - np.matmul(np.linalg.inv(R), (Y - targets))

        # Eq. 4.99 for Newton Raphson Gradient Descent
        w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
        w_new = np.matmul(w_new, iota.T)
        w_new = np.matmul(w_new, R)
        w_new = np.matmul(w_new, z)
        w_old = w_new

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
        data1 = data[0,counter]
        data2 = data[1,counter]
        data_a = np.vstack((data1, data2))
        if target == 1: # Class 1
            N1 += 1
            mu1 +=  targets[counter,:] * data_a # Eq. 4.75
        else: # Class 2
            N2 += 1
            mu2 +=  (1 - targets[counter,:]) * data_a # Eq. 4.76
    mu1 = (1/N1)*mu1 # Eq. 4.75
    mu2 = (1/N2)*mu2 # Eq. 4.76

    # Find covariance for classes (use same covariance for each)
    S1 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        data1 = data[0,counter]
        data2 = data[1,counter]
        data_a = np.vstack((data1, data2))
        if target == 1: # Class 1
            S1 += np.matmul((data_a - mu1), np.transpose(data_a - mu1)) # Eq. 4.79
    S1 = (1/N1) * S1 # Eq. 4.79

    S2 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        data1 = data[0,counter]
        data2 = data[1,counter]
        data_a = np.vstack((data1, data2))
        if target == 0: # Class 2
            S2 += np.matmul((data_a - mu2), np.transpose(data_a - mu2)) # Eq. 4.80
    S2 = (1/N2) * S2 # Eq. 4.80
    S = (N1/N)*S1 + (N2/N)*S2 # Eq. 4.78

    # Find the weights
    sigma = S
    # w0 is a prior, can set the initial probabilities how you desire, P_C1 is fed into the function from the main loop
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
    # print(divorce_targets.shape)

    # remove last column from original dataset and reshape into dimension M x N
    divorce_data = divorce_data[:,:-1]
    divorce_data = divorce_data.T
    # print(divorce_data.shape)
    
    # pick two quesitons that seem most relevant to make M = 2 instead of 54
    Q1 = 1 # Atr1 - If one of us apologizes when our discussion deteriorates, the discussion ends. 
    Q2 = 7 # Atr7 - We are like two strangers who share the same environment at home rather than family. 
    Q1_data = divorce_data[Q1-1, :]
    Q2_data = divorce_data[Q2-1, :]

    divorce_data = np.vstack((Q1_data, Q2_data))
    # print(divorce_data.shape)

    N = divorce_targets.shape[0]
    # print(N)

    P_C1 = 0.5
    divorce_w = getWeights(divorce_data, divorce_targets, P_C1)
    # print(divorce_w)

    # Classify the dataset based on the boundary line
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(divorce_targets):
        counter += 1
        p_class = sigmoid(np.matmul(divorce_w[1:,0].T, divorce_data[:,counter]) + divorce_w[0,0]) # Eq. 4.65
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == divorce_targets[counter,:]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('divorce', 'gaussian generative', success / (success + failure)))

    # Plot an ROC curve
    fig = plt.figure(1)
    plt.xlim(-0.01, 1)
    plt.ylim(0.0, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plots for Divorce Data')
    thresh = np.linspace(0, 1, 100)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    # Find the total number of positives and negatives in the data
    for target in np.nditer(divorce_targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1

    # Find number of false and true positives
    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(divorce_targets):
            counter += 1
            p_class = sigmoid(np.matmul(divorce_w[1:,0].T, divorce_data[:,counter]) + divorce_w[0,0]) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1
            else:
                target_guess = 0
            if target_guess == divorce_targets[counter,:] and divorce_targets[counter,:] == 1:
                TP += 1
            elif target_guess == 1 and divorce_targets[counter,:] == 0:
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'r', label = 'Gaussian Generative')
    plt.legend(loc='lower right')
    
    # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    divorce_w_irls = getWeightsIRLS(divorce_w[1:,0], divorce_data, divorce_targets)

    # Classify data with logistic regression model and report success
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(divorce_targets):
        counter += 1
        data1 = divorce_data[0,counter]
        data2 = divorce_data[1,counter]
        data_a = np.vstack((data1, data2))
        p_class = sigmoid(np.matmul(divorce_w_irls.T, data_a) + divorce_w[0,0]) # Eq. 4.65
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == divorce_targets[counter,0]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('divorce', 'logistic regression', success / (success + failure)))

    # Plot an ROC curve for divorce data with logistic regression
    fig = plt.figure(1)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    for target in np.nditer(divorce_targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1

    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(divorce_targets):
            counter += 1
            data1 = divorce_data[0,counter]
            data2 = divorce_data[1,counter]
            data_a = np.vstack((data1, data2))
            p_class = sigmoid(np.matmul(divorce_w_irls.T, data_a) + divorce_w[0,0]) # Eq. 4.65
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
    plt.plot(false_pos_rate, true_pos_rate, color = 'y', label='Logistic Regression')
    plt.legend(loc='lower right')

    plt.show()