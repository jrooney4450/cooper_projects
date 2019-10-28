import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat

def sigmoid(a):
    return (1 / (1 + np.exp(-a)))

def logisticRegression(w_old, phi, targets):
    N = targets.shape[1]
    # error_sum = 0
    counter = -1
    # for target in np.nditer(targets):
    #     counter += 1
    #     y = sigmoid(np.matmul(w.T, phi[:,counter])) # Eq. 4.91    
    #     error_sum += ((y[0] - targets[:,counter]) * phi[:,counter]) # Eq. 4.91

    # Construct the R matrix through Eq. 4.98
    R = np.eye(N)
    Y = np.zeros((N, 1))
    print(R.shape)
    for target in np.nditer(targets):
        counter += 1
        y = sigmoid(np.matmul(w_old.T, phi[:,counter])) # Eq. 4.91
        R[counter,counter] = y
        Y[counter,0] = y
    # print(y[0])
    # print(targets[:,counter][0])
    # print(targets.shape)
    # print(phi[:,counter])
    # print(error_sum)
    
    # Construct the iota matrix, N x M with rows of phi.T
    iota = phi.T[0]

    # Eq. 4.100 to construct Z matrix
    z = np.matmul(iota, w_old) - np.matmul(np.linalg.inv(R), (Y - target))

    # Eq. 4.99 for Newton Raphson Gradient Descent
    w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
    w_new = np.matmul(w_new, iota.T)
    w_new = np.matmul(w_new, R)
    w_new = np.matmul(w_new, z)
    print(w_new.shape)

def getWeights(data, targets):
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
    S2 = (1/N2) * S2
    S = (N1/N)*S1 + (N2/N)*S2 # Eq. 4.80
    # print(S)

    # Find the weights
    sigma = S
    # w0 is a prior, can set the probabilities how you desire
    p_C1 = 0.5
    p_C2 = 1 - p_C1
    w0 = -0.5 * np.matmul(np.matmul(mu1.T, np.linalg.inv(sigma)), mu1) + 0.5 * np.matmul(np.matmul(mu2.T, np.linalg.inv(sigma)), mu2) \
        + np.log(p_C1 / p_C2) # Eq. 4.67
    w0 = w0[0,0]
    w = np.matmul(np.linalg.inv(sigma), mu1 - mu2) # Eq. 4.66
    w = np.array([[w0],[w[0,0]],[w[1,0]]])
    # print(w.shape)
    return w

if __name__ == "__main__":

    data = loadmat('mlData.mat')

    unimodal_data = np.array([data['unimodal'][0][0][0]])
    unimodal_data = unimodal_data.T # Transpose to make x into vectors - M X N
    unimodal_targets = np.array([data['unimodal'][0][0][1]])
    # print(unimodal_targets)

    # print(unimodal_data.shape)

    # Initialize raw data plot
    fig = plt.figure(1)
    plt.xlim(-4,6)
    plt.ylim(-5,6)
    counter = -1

    # Plot points based on the target value
    for target in np.nditer(unimodal_targets):
        counter += 1
        if target == 1:
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], marker='x', color = 'r') # Class 1, t = 1
        else:
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 0

    # count the size of the data

    
    # print(unimodal_data[:,0]) # This one works! Don't change
    # print(np.matmul(unimodal_data[:,0], np.transpose(unimodal_data[:,0])))

    # train_ratio = 0.8
    # N_train = int(0.8*N)
    # unimodal_data_train = unimodal_data[0:N_train,:]
    # unimodal_targets_train = unimodal_data[0:N_train,:]
    
    unimodal_w = getWeights(unimodal_data, unimodal_targets)
    # print(unimodal_w)

    logisticRegression(unimodal_w[1:,0], unimodal_data, unimodal_targets)

    # y = np.matmul(w.T, x) # what is x here?

    # Plot the boundary between the classes
    x = np.linspace(-3,5,100)
    # x2 = np.linspace(-3,5,100)
    fig = plt.figure(1)
    plt.plot(x, unimodal_w[1,0]*x + unimodal_w[2,0]*x + unimodal_w[0,0], color = 'g')

    # Classify a random point based on the training set
    point = np.array([[1.5],[0]])
    # print(point[0,0])
    fig = plt.figure(1)
    plt.scatter(point[0,0], point[1,0], color = 'orange')
    classify = sigmoid(np.matmul(unimodal_w[1:,0].T, point) + unimodal_w[0,0])
    threshold = 0.5
    if classify[0] >= threshold:
        classify = 1
    else:
        classify = 0
    # print("The point ({}, {}) is part of class {}".format(point[0,0], point[1,0], classify))

    # logisticRegression()

    ########################### Circles stuff

    circles_data = np.array([data['circles'][0][0][0]])
    circles_data = circles_data.T
    circles_targets = np.array([data['circles'][0][0][1]])

    fig = plt.figure(2)
    counter = -1

    # Plot points based on the target value
    for target in np.nditer(unimodal_targets):
        counter += 1
        if target == 1:
            plt.scatter(circles_data[0,counter], circles_data[1,counter], marker='x', color = 'r') # Class 1, t = 1
        else:
            plt.scatter(circles_data[0,counter], circles_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 0

    circles_w = getWeights(circles_data, circles_targets)

    # # Add additional basis function for circles data x3 = x1^2 + x2^3
    # circles_phi = np.zeros((1, N_circles, 1))
    # counter = -1
    # for target in np.nditer(circles_targets):
    #     counter += 1
    #     circles_phi[0,counter] = (circles_data[0,counter]**2 + circles_data[1,counter]**2)
    # # circles_phi = np.vstack(circles_data, circles_phi)
    # circles_phi = np.concatenate((circles_data, circles_phi), axis=0)
    # print(circles_phi.shape)
    # print(circles_phi[:,1])

    # plt.show()