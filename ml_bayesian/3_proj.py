import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat

def sigmoid(a):
    return (1 / (1 + np.exp(-a)))

def getWeightsIRLS(w_old, phi, targets):
    N = targets.shape[1]

    # Construct the R matrix through Eq. 4.98
    R = np.eye(N)
    Y = np.zeros((N, 1))

    # print(Y.shape)
    # print(w_old.shape)

    # Construct the iota matrix, N x M with rows of phi.T
    iota = phi.T[0]
    difference = 1

    # IRLS Algorithm - complete when the difference between the data points is very small
    while difference >= 0.00005:
        counter = -1
        for target in np.nditer(targets):
            counter += 1
            y = sigmoid(np.matmul(w_old.T, phi[:,counter])) # Eq. 4.91
            R[counter,counter] = y
            Y[counter,0] = y

        # Eq. 4.100 to construct Z matrix
        z = np.matmul(iota, w_old) 
        z = z - np.matmul(np.linalg.inv(R), (Y - targets[0]))

        # Eq. 4.99 for Newton Raphson Gradient Descent
        w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
        w_new = np.matmul(w_new, iota.T)
        w_new = np.matmul(w_new, R)
        w_new = np.matmul(w_new, z)
        difference = np.abs(w_old[0,0] - w_new[0,0])
        w_old = w_new
        # print(w_new)
        # print(difference)

    return w_new

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
    S2 = (1/N2) * S2 # Eq. 4.80
    S = (N1/N)*S1 + (N2/N)*S2 # Eq. 4.78

    # Find the weights
    sigma = S
    # w0 is a prior, can set the initial probabilities how you desire
    p_C1 = 0.5
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
    unimodal_targets = np.array([data['unimodal'][0][0][1]])

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
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], marker='x', color = 'r') # Class 1, t = 1
        else:
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 0

    unimodal_w = getWeights(unimodal_data, unimodal_targets)
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
        p_class = sigmoid(np.matmul(unimodal_w[1:,0].T, unimodal_data[:,counter]) + unimodal_w[0,0]) #
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == unimodal_targets[:,counter]:
            success += 1
        else:
            failure += 1
    
    print("The percentage of successfully classified {} data is: {}".format('unimodal', success / (success + failure)))

    # Apply a logisitic regression using the ML weights
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
    
    print("The percentage of successfully classified {} data is: {}".format('unimodal irls', success / (success + failure)))

    

    ########################### Circles!!!

    circles_data = np.array([data['circles'][0][0][0]])
    circles_data = circles_data.T
    circles_targets = np.array([data['circles'][0][0][1]])

    # Initialize a figure to plot circles data
    fig = plt.figure(2)
    plt.xlim(-1,3)
    plt.ylim(-1,3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Classification of Circles Data')
    counter = -1

    # Plot points based on the target value
    for target in np.nditer(unimodal_targets):
        counter += 1
        if target == 1:
            plt.scatter(circles_data[0,counter], circles_data[1,counter], marker='x', color = 'r') # Class 1, t = 1
        else:
            plt.scatter(circles_data[0,counter], circles_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 0

    # Get the ML estimate on the weights
    circles_w = getWeights(circles_data, circles_targets)

    # Plot the boundary between the classes
    fig = plt.figure(2)
    # print(circles_w)
    plt.plot(x, (-circles_w[1,0]*x - circles_w[0,0]) / circles_w[2,0], color = 'g', label='Gaussian Generative')
    plt.legend(loc='lower right')

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
    
    # print("The percentage of successfully classified data {} is: {}".format('circles', success / (success + failure)))

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

    plt.show()