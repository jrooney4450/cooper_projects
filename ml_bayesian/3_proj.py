import math
import numpy as np
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat

def sigmoid(a):
    return (1 / (1 + np.exp(-a)))

def getWeightsIRLS(w_old, phi, targets):
    N = targets.shape[1]

    # Construct the iota matrix (N x M) with rows of phi.T
    iota = phi.T[0]
    # print(iota.shape)

    # Initialize parameters that will be updated in IRLS loop
    R = np.eye(N)
    Y = np.zeros((N, 1))

    # IRLS Algorithm - complete when the difference between the data points is very small
    for i in range(100):
    # while difference > 0.0001: # option to use a step size to stop the loop
        counter = -1

        # Construct Y matrix (N x 1) and R matrix (N x N), Eq. 4.98
        for target in np.nditer(targets):
            counter += 1
            y = sigmoid(np.matmul(w_old.T, phi[:,counter])) # Eq. 4.91
            R[counter,counter] = y
            Y[counter,0] = y
        print(np.linalg.inv(R))

        # Eq. 4.100 to construct Z matrix
        z = np.matmul(iota, w_old)
        dummy = Y - targets[0]
        z = z - np.matmul(np.linalg.inv(R), dummy)

        # Eq. 4.99 for Newton Raphson Gradient Descent
        w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
        w_new = np.matmul(w_new, iota.T)
        w_new = np.matmul(w_new, R)
        w_new = np.matmul(w_new, z)
        # difference = np.abs(w_old[1,0] - w_new[1,0]) # This is how the while loop checks step size
        w_old = w_new

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
        else: # Class 2
            N2 += 1
            mu2 +=  (1 - targets[:,counter]) * data[:,counter] # Eq. 4.76
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
    S2 = (1/N2) * S2 # Eq. 4.80
    S = (N1/N)*S1 + (N2/N)*S2 # Eq. 4.78

    # Find the weights
    sigma = S
    p_C2 = 1 - p_C1
    w0 = -0.5 * np.matmul(np.matmul(mu1.T, np.linalg.inv(sigma)), mu1) + 0.5 * np.matmul(np.matmul(mu2.T, np.linalg.inv(sigma)), mu2) \
        + np.log(p_C1 / p_C2) # Eq. 4.67
    # w0 = w0[0,0]
    w = np.matmul(np.linalg.inv(sigma), mu1 - mu2) # Eq. 4.66
    w = np.vstack((w0,w)) # store weights into a vector
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
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], marker='x', color = 'r') # Class 1, t = 0
        else:
            plt.scatter(unimodal_data[0,counter], unimodal_data[1,counter], facecolors='none', edgecolors='blue') # Class 2, t = 1

    # Find the ML estimate of the weights using a prior based on the probability that a target is in class 1
    P_C1 = 0.5
    unimodal_w = getWeights(unimodal_data, unimodal_targets, P_C1)

    # Plot the boundary between the classes
    x = np.linspace(-5,6,100)
    fig = plt.figure(1)
    plt.plot(x, (-unimodal_w[1,0]*x - unimodal_w[0,0]) / unimodal_w[2,0], color = 'g', label = 'Gaussian Generative Boundary')
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

    # Initialize plot and parameters for ROC plots - this one will handle gaussian generative for the unimodal data
    fig = plt.figure(2)
    plt.xlim(-0.01, 1)
    plt.ylim(0.0, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Plots')
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

    # Find number of false and true positives
    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(unimodal_targets):
            counter += 1
            p_class = sigmoid(np.matmul(unimodal_w[1:,0].T, unimodal_data[:,counter]) + unimodal_w[0,0]) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1
            else:
                target_guess = 0
            if target_guess == unimodal_targets[:,counter] and unimodal_targets[:,counter] == 1: # Find and sum true positives
                TP += 1
            elif target_guess == 1 and unimodal_targets[:,counter] == 0: # Find and sum false positives
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'r', label = 'Unimodal Data Logistic Regression')
    plt.legend(loc='lower right')
    
    ################################# Logistic Regression ####################################

    # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    unimodal_w_irls = getWeightsIRLS(unimodal_w[1:,:], unimodal_data, unimodal_targets)

    fig = plt.figure(1)
    plt.plot(x, (-unimodal_w_irls[0,0]*x - unimodal_w[0,0]) / unimodal_w_irls[1,0], color = 'orange', label = 'Logistic Regression Boundary')
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

    # Plot an ROC curve for logisitc regressuion of the unimodal data
    fig = plt.figure(2)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1
    # Search for true positives and false positives based on a range of all possible thresholds
    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(unimodal_targets):
            counter += 1
            p_class = sigmoid(np.matmul(unimodal_w_irls.T, unimodal_data[:,counter])) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1
            else:
                target_guess = 0
            if target_guess == unimodal_targets[:,counter] and unimodal_targets[:,counter] == 1: # Find and sum true positives
                TP += 1
            elif target_guess == 1 and unimodal_targets[:,counter] == 0: # Find and sum false positives
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'y', label = 'Unimodal Data Logistic Regression')
    plt.legend('lower right')

    ###########################################################################################
    ################################### Circles!!! ############################################
    ###########################################################################################

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

    # Add additional basis function for circles data, phi_3 = x1^2 + x2^3
    circles_phi = np.zeros((1, N_circles, 1))
    counter = -1
    for target in np.nditer(circles_targets):
        counter += 1
        circles_phi[0,counter] = (circles_data[0,counter]**2 + circles_data[1,counter]**2)
    circles_phi = np.vstack((circles_data, circles_phi))
    # print(circles_phi.shape)

    # Find the ML estimate of the weights using a prior based on the probability that a target is in class 1
    P_C1 = 0.5
    circles_w = getWeights(circles_phi, circles_targets, P_C1)
    # print(circles_w)

    # print(circles_phi.shape)

    # Classify data with gaussian generative model and report success
    threshold = 0.5
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(circles_targets):
        counter += 1
        p_class = sigmoid(np.matmul(circles_w[1:,0].T, circles_phi[:,counter]) + circles_w[0,0])
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == circles_targets[:,counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} data using {} is: {}".format('circles', 'gaussian generative', success / (success + failure)))

    # Plot an ROC curve for the gaussian generative classifier for the circles data
    fig = plt.figure(2)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1

    # Find the total number of positives and negatives in the data
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    for target in np.nditer(circles_targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1
    
    # Find the number of true and false positives
    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(circles_targets):
            counter += 1
            p_class = sigmoid(np.matmul(circles_w[1:,0].T, circles_phi[:,counter]) + circles_w[0,0]) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1 
            else:
                target_guess = 0
            if target_guess == circles_targets[:,counter] and circles_targets[:,counter] == 1: # Find and sum true positives
                TP += 1
            elif target_guess == 1 and circles_targets[:,counter] == 0: # Find and sum false positives
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'g', label = 'Cicles Data Gaussian Generative')
    plt.legend(loc='lower right')

    ################################# Logistic Regression ####################################

    # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    circles_w_irls = getWeightsIRLS(circles_w[1:,:], circles_phi, circles_targets)

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

    # Plot an ROC curve for the logistic regression of the circles data
    fig = plt.figure(2)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1
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
            if target_guess == circles_targets[:,counter] and circles_targets[:,counter] == 1: # Find and sum true positives
                TP += 1
            elif target_guess == 1 and circles_targets[:,counter] == 0: # Find and sum false positives
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = 'b', label = 'Cicles Data Logistic Regression')
    plt.legend(loc='lower right')

    plt.show()