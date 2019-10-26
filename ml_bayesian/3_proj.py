import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat

def sigmoid(a):
    return (1 / (1 + np.exp(-a)))

def logisticRegression(N, data, targets):



def gaussianGenerative(N, data, targets):
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
    mu2 = (1/N2)*mu2 # Eq. 4.76

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
    p_C1 = 0.499
    p_C2 = 1 - p_C1
    w0 = -0.5 * np.matmul(np.matmul(mu1.T, np.linalg.inv(sigma)), mu1) + 0.5 * np.matmul(np.matmul(mu2.T, np.linalg.inv(sigma)), mu2) \
        + np.log(p_C1 / p_C2) # Eq. 4.67
    
    w = np.matmul(np.linalg.inv(sigma), mu1 - mu2) # Eq. 4.66
    # print(w0)
    # print(w)

    # Classify a random point based on training set
    x = np.linspace(-5,5,100)
    fig = plt.figure(1)
    point = np.array([[3],[3]])

    classify = sigmoid(np.matmul(w.T, point) + w0)
    threshold = 0.5
    if classify[0] >= threshold:
        classify = 1
    else:
        classify = 0
    print("The point ({}, {}) is part of class {}".format(point[0,0], point[1,0], classify))


if __name__ == "__main__":

    data = loadmat('mlData.mat')

    unimodal_data = np.array([data['unimodal'][0][0][0]])
    unimodal_data = unimodal_data.T # Transpose to make x into vectors - M X N
    unimodal_targets = np.array([data['unimodal'][0][0][1]])
    fig = plt.figure(1)
    plt.scatter(unimodal_data[0,:], unimodal_data[1,:])

    # count the size of the data
    N = unimodal_targets.shape[1]
    # print(N)
    
    # print(unimodal_data[:,0]) # This one works! Don't change
    # print(np.matmul(unimodal_data[:,0], np.transpose(unimodal_data[:,0])))

    # print(unimodal_data.T[:,0].reshape((-1,1)))
    # print(unimodal_data.shape)

    # print(unimodal_targets[:,0])

    # train_ratio = 0.8
    # N_train = int(0.8*N)
    # unimodal_data_train = unimodal_data[0:N_train,:]
    # unimodal_targets_train = unimodal_data[0:N_train,:]
    gaussianGenerative(N, unimodal_data, unimodal_targets)

    logisticRegression(N, unimodal_data, unimodal_targets)

    # TODO: find mu1 and mu2 by 4.75 and 4.76


    # circles_data = data['circles'][0][0][0]
    # circles_tragets = data['circles'][0][0][1]
    # fig = plt.figure()
    # plt.scatter(circles_data[:,0], circles_data[:,1])

    # plt.show()