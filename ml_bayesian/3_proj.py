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
    N = targets.shape[1]
    # print(phi.shape)

    # Construct the iota matrix (N x M) with rows of phi.T
    iota = phi.T[0]
    # print(iota.shape)

    # Initialize parameters that will be updated in IRLS loop
    R = np.eye(N)
    # print(R.shape)
    Y = np.zeros((N, 1))
    # difference = 1

    # print(targets[0].shape)

    # IRLS Algorithm - complete when the difference between the data points is very small
    for i in range(100):
    # while difference > 0.0001:
        counter = -1

        # Construct Y matrix (N x 1) and R matrix (N x N), Eq. 4.98
        for target in np.nditer(targets):
            counter += 1
            y = sigmoid(np.matmul(w_old.T, phi[:,counter])) # Eq. 4.91
            R[counter,counter] = y
            Y[counter,0] = y

        # Eq. 4.100 to construct Z matrix
        z = np.matmul(iota, w_old) - np.matmul(np.linalg.inv(R), (Y - targets[0]))

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
    N = targets.shape[1]

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

    data = loadmat('mlData.mat')

    unimodal_data = np.array([data['unimodal'][0][0][0]])
    unimodal_data = unimodal_data.T # Transpose to make x into vectors - M X N
    # print(unimodal_data.shape)
    unimodal_targets = np.array([data['unimodal'][0][0][1]])
    # print(unimodal_targets.shape)

    # Initialize raw data plot
    fig = plt.figure(1)
    plt.xlim(-4,6)
    plt.ylim(-5,6)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Classification of Unimodal Data')
    counter = -1

    # Plot points based on the target value
    for target in np.nditer(unimodal_targets):
        counter += 1
        if target == 1:
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], marker='x', color = 'r') # Class 1, t = 0
            # plt.legend(loc='lower right')
        else:
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 1

    # Find the ML estimate of the weights using a prior based on the probability that a target is in class 1
    P_C1 = 0.5
    unimodal_w = getWeights(unimodal_data, unimodal_targets, P_C1)
    # print(unimodal_w)

    # Plot the boundary between the classes
    x = np.linspace(-5,6,100)
    fig = plt.figure(1)
    # plt.legend(loc='lower right')
    # print(unimodal_w)
    plt.plot(x, (-unimodal_w[1,0]*x - unimodal_w[0,0]) / unimodal_w[2,0], color = 'g', label = 'Gaussian Generative')
    plt.legend(loc='lower right')

    # Classify the dataset based on the boundary line
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(unimodal_targets):
        counter += 1
        p_class = sigmoid(np.matmul(unimodal_w[1:,0].T, unimodal_data[:,counter]) + unimodal_w[0,0]) # Eq. 4.65
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == unimodal_targets[:,counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('unimodal', 'gaussian generative', success / (success + failure)))

    # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    unimodal_w_irls = getWeightsIRLS(unimodal_w[1:,:], unimodal_data, unimodal_targets)

    fig = plt.figure(1)
    plt.plot(x, (-unimodal_w_irls[0,0]*x - unimodal_w[0,0]) / unimodal_w_irls[1,0], color = 'orange', label = 'Logistic Regression')
    plt.legend(loc='lower right')

    # Classify the dataset based on the boundary line
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(unimodal_targets):
        counter += 1
        p_class = sigmoid(np.matmul(unimodal_w_irls[:,0].T, unimodal_data[:,counter]) + unimodal_w[0,0]) #
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == unimodal_targets[:,counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('unimodal', 'logistic regression', success / (success + failure)))

    # Initialize a figure to plot an ROC curve for unimodal data
    fig = plt.figure(2)
    plt.xlim(-0.01, 1)
    plt.ylim(0.0, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot for Unimodal Data')

    # Formulate the ROC curve
    thresh = np.linspace(0, 1, 100)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    # Find the total number of positives and negatives in the data
    for target in np.nditer(unimodal_targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1

    # Search for true positives and false positives based on a range of all possible thresholds
    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(unimodal_targets):
            counter += 1
            # p_class = sigmoid(np.matmul(circles_w_irls[:,0].T, circles_phi[:,counter]) + circles_w[0,0])
            p_class = sigmoid(np.matmul(unimodal_w_irls.T, unimodal_data[:,counter])) # Eq. 
            if p_class >= thresh[i]:
                target_guess = 1
            else:
                target_guess = 0
            if target_guess == unimodal_targets[:,counter] and unimodal_targets[:,counter] == 1:
                TP += 1
            elif target_guess == 1 and unimodal_targets[:,counter] == 0:
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'g')

    # ##########################################################################################
    # ################################## Circles!!! ############################################
    # ##########################################################################################

    circles_data = np.array([data['circles'][0][0][0]])
    circles_data = circles_data.T
    circles_targets = np.array([data['circles'][0][0][1]])

    N_circles = circles_targets.shape[1]

    # Initialize a figure to plot circles data
    fig = plt.figure(3)
    plt.xlim(-1,3)
    plt.ylim(-1,3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Classification of Circles Data')
    counter = -1

    # Plot points based on the target value
    for target in np.nditer(circles_targets):
        counter += 1
        if target == 1:
            plt.scatter(circles_data[0,counter], circles_data[1,counter], marker='x', color = 'r') # Class 1, t = 1
        else:
            plt.scatter(circles_data[0,counter], circles_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 0

    # Find the ML estimate of the weights using a prior based on the probability that a target is in class 1
    P_C1 = 0.5
    circles_w = getWeights(circles_data, circles_targets, P_C1)

    # Classify data with gaussian generative model and report success
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(circles_targets):
        counter += 1
        p_class = sigmoid(np.matmul(circles_w[1:,0].T, circles_data[:,counter]) + circles_w[0,0])
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == circles_targets[:,counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('circles', 'gaussian generative', success / (success + failure)))

    # Add additional basis function for circles data, phi_3 = x1^2 + x2^3
    circles_phi = np.zeros((1, N_circles, 1))
    counter = -1
    for target in np.nditer(circles_targets):
        counter += 1
        circles_phi[0,counter] = (circles_data[0,counter]**2 + circles_data[1,counter]**2)
    circles_phi = np.vstack((circles_data, circles_phi))
    # circles_phi = np.concatenate((circles_data, circles_phi), axis=0)
    # print(circles_phi.shape)
    # print(circles_phi[:,100])

    # need to kick off IRLS with an addiiotnal weight to account for addded third basis function
    add_w = np.array([1])
    # print(circles_w.shape)
    circles_w = np.vstack((circles_w, add_w))
    # print(circles_w)
    # print(circles_w[1:,:])
    # print(circles_w[0,0])

    # print(circles_targets.shape)
    # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    circles_w_irls = getWeightsIRLS(circles_w[1:,:], circles_phi, circles_targets)
    # print(circles_w_irls)

    # Classify data with gaussian generative model and report success
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(circles_targets):
        counter += 1
        p_class = sigmoid(np.matmul(circles_w_irls.T, circles_phi[:,counter]) + circles_w[0,0]) # Eq. 4.65
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == circles_targets[:,counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('circles', 'logistic regression', success / (success + failure)))

    # Initialize a figure to plot an ROC curve for circles data
    fig = plt.figure(4)
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plot for Circles Data')

    thresh = np.linspace(0, 1, 100)
    true_pos_rate = []
    false_pos_rate = []
    # threshold = 0.4
    counter = -1
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    for target in np.nditer(circles_targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1

    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(circles_targets):
            counter += 1
            p_class = sigmoid(np.matmul(circles_w_irls.T, circles_phi[:,counter]) + circles_w[0,0]) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1 
            else:
                target_guess = 0
            if target_guess == circles_targets[:,counter] and circles_targets[:,counter] == 1:
                TP += 1
            elif target_guess == 1 and circles_targets[:,counter] == 0:
                FP += 1
        # if P == 0 or N == 0:
        #     true_pos_rate.append(0)
        #     false_pos_rate.append(0)
        # else:
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'g')

    ##########################################################################################
    ################################## Divorce!!! ############################################
    ##########################################################################################

    # Data obtained through UCI, see here https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
    # divorce_data = pd.read_csv("divorce.csv")
    # print(type(divorce_data))
    
    # data_path = 'divorce.csv'

    # # The data has features of M = 54 with total datapoints N = 170
    # divorce_data = np.loadtxt(data_path, delimiter=';', skiprows=1)

    # # extract the targets in the last column and convert to dimension N x 1
    # divorce_targets = divorce_data[:,-1]
    # divorce_targets = np.array([divorce_targets])
    # divorce_targets = np.array([divorce_targets.T])
    # # print(divorce_targets[0].shape)

    # # remove last column from original dataset and reshape into dimension M x N
    # divorce_data = divorce_data[:,:-1]
    # divorce_data = np.array([divorce_data.T])
    # # print(divorce_data[0].shape)

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

    # # Classify data with gaussian generative model and report success
    # threshold = 0.5
    # success = 0
    # failure = 0
    # counter = -1
    # for target in np.nditer(circles_targets):
    #     counter += 1
    #     p_class = sigmoid(np.matmul(circles_w[1:,0].T, circles_data[:,counter]) + circles_w[0,0])
    #     if p_class >= threshold:
    #         target_guess = 1
    #     else:
    #         target_guess = 0
    #     if target_guess == circles_targets[:,counter]:
    #         success += 1
    #     else:
    #         failure += 1
    # print("The percentage of successfully classified {} data using {} is: {}".format('circles', 'gaussian generative', success / (success + failure)))

    # # Add additional basis function for circles data, phi_3 = x1^2 + x2^3
    # circles_phi = np.zeros((1, N_circles, 1))
    # counter = -1
    # for target in np.nditer(circles_targets):
    #     counter += 1
    #     circles_phi[0,counter] = (circles_data[0,counter]**2 + circles_data[1,counter]**2)
    # circles_phi = np.vstack((circles_data, circles_phi))
    # # circles_phi = np.concatenate((circles_data, circles_phi), axis=0)
    # # print(circles_phi.shape)
    # # print(circles_phi[:,100])

    # # need to kick off IRLS with an addiiotnal weight to account for addded third basis function
    # add_w = np.array([1])
    # # print(circles_w.shape)
    # circles_w = np.vstack((circles_w, add_w))
    # # print(circles_w)
    # # print(circles_w[1:,:])
    # # print(circles_w[0,0])

    # # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    # divorce_w = np.ones((54, 1))
    # # print(divorce_w.shape)
    # circles_w_irls = getWeightsIRLS(divorce_w, divorce_data, divorce_targets)
    # print(circles_w_irls)

    # # Initialize a figure to plot an ROC curve for circles data
    # fig = plt.figure(6)
    # plt.xlim(-0.01, 1)
    # plt.ylim(0, 1.01)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Plot for Circles Data')

    # thresh = np.linspace(0, 1, 100)
    # true_pos_rate = []
    # false_pos_rate = []
    # # threshold = 0.4
    # counter = -1
    # P = 0 # number of positives in data (when t = 1)
    # N = 0 # number of negatives in data (when t = 0)
    # for target in np.nditer(circles_targets):
    #     counter += 1
    #     if target == 1: # Class 1
    #         P += 1
    #     else: # Class 2
    #         N += 1

    # for i in range(len(thresh)):
    #     FP = 0 # number of false positives
    #     TP = 0 # number of true positives
    #     counter = -1
    #     for target in np.nditer(circles_targets):
    #         counter += 1
    #         p_class = sigmoid(np.matmul(circles_w_irls.T, circles_phi[:,counter]) + circles_w[0,0]) # Eq. 4.65
    #         if p_class >= thresh[i]:
    #             target_guess = 1 
    #         else:
    #             target_guess = 0
    #         if target_guess == circles_targets[:,counter] and circles_targets[:,counter] == 1:
    #             TP += 1
    #         elif target_guess == 1 and circles_targets[:,counter] == 0:
    #             FP += 1
    #     # if P == 0 or N == 0:
    #     #     true_pos_rate.append(0)
    #     #     false_pos_rate.append(0)
    #     # else:
    #     true_pos_rate.append(TP / P)
    #     false_pos_rate.append(FP / N)
    # plt.plot(false_pos_rate, true_pos_rate, color = 'g')

    # print("The percentage of successfully classified {} data using {} is: {}".format('circles', 'logistic regression', success / (success + failure)))

    plt.show()