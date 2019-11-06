import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat
import csv
import sklearn
from sklearn import svm

def sigmoid(a):
    return (1 / (1 + np.exp(-a)))

def getWeightsIRLS(w_old, data, targets):
    # Format weights into vector
    # w1 = w_old[0]
    # w2 = w_old[1]
    # w_old = np.vstack((w1, w2))
    M = data.shape[0]
    # print(M)
    w_old = w_old[:M,:]
    # print(w_old)

    N = targets.shape[0]

    # Construct the iota matrix (N x M) with rows of data.T
    iota = data.T

    # Initialize parameters that will be updated in IRLS loop
    R = np.eye(N)
    Y = np.zeros((N, 1))

    # IRLS Algorithm - complete a set amount of loops
    for i in range(100):
        counter = -1

        # Construct Y matrix (N x 1) and R matrix (N x N), Eq. 4.98
        for target in np.nditer(targets):
            counter += 1
            y = sigmoid(np.matmul(w_old.T, data[:,counter])) # Eq. 4.91
            R[counter,counter] = y
            Y[counter,0] = y

        w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
        w_new = np.matmul(w_new, iota.T)
        w_new = np.matmul(w_new, Y - targets)
        w_new = w_old - w_new

        # # Eq. 4.100 to construct Z matrix
        # z = np.matmul(iota, w_old)
        # z = z - np.matmul(np.linalg.inv(R), (Y - targets))

        # # Eq. 4.99 for Newton Raphson Gradient Descent
        # w_new = np.linalg.inv(np.matmul(np.matmul(iota.T, R), iota))
        # w_new = np.matmul(w_new, iota.T)
        # w_new = np.matmul(w_new, R)
        # w_new = np.matmul(w_new, z)
        # w_old = w_new

    return w_new

def getWeights(data, targets, p_C1):
    N = targets.shape[0]

    # Find ML mean for each class
    N1 = 0
    N2 = 0
    mu1 = 0
    mu2 = 0
    counter = -1

    # print(data.shape)
    # print(data[:,0].reshape(-1,1))
    for target in np.nditer(targets):
        counter += 1
        # data1 = data[0,counter]
        # data2 = data[1,counter]
        # data_a = np.vstack((data1, data2))
        if target == 1: # Class 1
            N1 += 1
            mu1 +=  targets[counter,:] * data[:,counter].reshape(-1,1) # Eq. 4.75
        else: # Class 2
            N2 += 1
            mu2 +=  (1 - targets[counter,:]) * data[:,counter].reshape(-1,1) # Eq. 4.76
    mu1 = (1/N1)*mu1 # Eq. 4.75
    # print(mu1)
    mu2 = (1/N2)*mu2 # Eq. 4.76
    # print(mu2)

    # Find covariance for classes (use same covariance for each)
    S1 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        # data1 = data[0,counter]
        # data2 = data[1,counter]
        # data_a = np.vstack((data1, data2))
        if target == 1: # Class 1
            S1 += np.matmul((data[:,counter].reshape(-1,1) - mu1), np.transpose(data[:,counter].reshape(-1,1) - mu1)) # Eq. 4.79
    S1 = (1/N1) * S1 # Eq. 4.79

    S2 = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        # data1 = data[0,counter]
        # data2 = data[1,counter]
        # data_a = np.vstack((data1, data2))
        if target == 0: # Class 2
            S2 += np.matmul((data[:,counter].reshape(-1,1) - mu2), np.transpose(data[:,counter].reshape(-1,1) - mu2)) # Eq. 4.80
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
    # print('the weights from the logistic regression are {}'.format(w))
    w = np.vstack((w, w0))
    return w

def classify(data, targets, weights, w0, threshold, model):
    # Classify the dataset based on the boundary line
    M = data.shape[0]
    weights = weights[:M,:]
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(targets):
        counter += 1
        p_class = sigmoid(np.matmul(weights.T, data[:,counter]) + w0) # Eq. 4.65
        if p_class >= threshold:
            target_guess = 1
        else:
            target_guess = 0
        if target_guess == targets[counter,:]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} is: {}".format(model, success / (success + failure)))

def plotROC(data, targets, weights, w0, label, color):
    # Plot an ROC curve for the gaussian generative classifier for the circles data
    fig = plt.figure(1)
    true_pos_rate = []
    false_pos_rate = []
    counter = -1
    thresh = np.linspace(0, 1, 100)
    M = data.shape[0]
    weights = weights[:M,:]

    # Find the total number of positives and negatives in the data
    P = 0 # number of positives in data (when t = 1)
    N = 0 # number of negatives in data (when t = 0)
    for target in np.nditer(targets):
        counter += 1
        if target == 1: # Class 1
            P += 1
        else: # Class 2
            N += 1

    # print(targets[:,0])

    # Find the number of true and false positives
    for i in range(len(thresh)):
        FP = 0 # number of false positives
        TP = 0 # number of true positives
        counter = -1
        for target in np.nditer(targets):
            counter += 1
            p_class = sigmoid(np.matmul(weights.T, data[:,counter]) + w0) # Eq. 4.65
            if p_class >= thresh[i]:
                target_guess = 1 
            else:
                target_guess = 0
            if target_guess == targets[counter,:] and targets[counter,:] == 1: # Find and sum true positives
                TP += 1
            elif target_guess == 1 and targets[counter,:] == 0: # Find and sum false positives
                FP += 1
        true_pos_rate.append(TP / P)
        false_pos_rate.append(FP / N)
    plt.plot(false_pos_rate, true_pos_rate, color = color, label = label)
    plt.legend(loc='lower right')

# class SVM:
#     def fit(self, X, y):
#         n_samples, n_features = X.shape # P = X^T X
#         K = np.zeros((n_samples, n_samples))
#         for i in range(n_samples):
#             for j in range(n_samples):
#                 K[i,j] = np.dot(X[i], X[j])P = cvxopt.matrix(np.outer(y, y) * K)# q = -1 (1xN)
#         q = cvxopt.matrix(np.ones(n_samples) * -1)# A = y^T 
#         A = cvxopt.matrix(y, (1, n_samples))# b = 0 
#         b = cvxopt.matrix(0.0)# -1 (NxN)
#         G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))# 0 (1xN)
#         h = cvxopt.matrix(np.zeros(n_samples))solution = cvxopt.solvers.qp(P, q, G, h, A, b)# Lagrange multipliers
#         a = np.ravel(solution['x'])# Lagrange have non zero lagrange multipliers
#         sv = a > 1e-5
#         ind = np.arange(len(a))[sv]
#         self.a = a[sv]
#         self.sv = X[sv]
#         self.sv_y = y[sv]# Intercept
#         self.b = 0
#         for n in range(len(self.a)):
#             self.b += self.sv_y[n]
#             self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
#         self.b /= len(self.a)# Weights
#         self.w = np.zeros(n_features)
#         for n in range(len(self.a)):
#             self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        
#     def project(self, X):
#         return np.dot(X, self.w) + self.b
        
#     def predict(self, X):
#         return np.sign(self.project(X))


if __name__ == "__main__":

    data = loadmat('mlData.mat')

    circles_targets = data['circles'][0][0][1]
    # print(circles_targets.shape)

    circles_data = data['circles'][0][0][0]
    circles_data = circles_data.T
    # print(circles_data.shape)

    N = circles_targets.shape[0]
    # print(N)

    # Add additional basis function for circles data, phi_3 = x1^2 + x2^3
    circles_phi = np.zeros((1, N))
    counter = -1
    for target in np.nditer(circles_targets):
        counter += 1
        circles_phi[0,counter] = (circles_data[0,counter]**2 + circles_data[1,counter]**2)
    circles_data = np.vstack((circles_data, circles_phi))
    print(circles_data.shape)

    # fig = plt.figure(1)
    # plt.xlim(-0.01, 1)
    # plt.ylim(0.0, 1.01)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Plots')

    # # Do the things on the circles data
    # P_C1 = 0.5

    # circles_w = getWeights(circles_data, circles_targets, P_C1)
    # circles_w0 = circles_w[-1,0]
    # classify(circles_data, circles_targets, circles_w, circles_w0, 0.5, 'circles data using gaussian generative')
    # plotROC(circles_data, circles_targets, circles_w, circles_w0, 'circles data gaussian generative', 'r')

    # # w = np.ones((3,1))
    # circles_w_irls = getWeightsIRLS(circles_w, circles_data, circles_targets)
    # classify(circles_data, circles_targets, circles_w_irls, circles_w0, 0.5, 'circles data using logistic regression')
    # plotROC(circles_data, circles_targets, circles_w_irls, circles_w0, 'circles data logistic regression', 'y')

    # Remove third basis function to test SVM
    circles_data = circles_data[:-1,:]
    print(circles_data.shape)

    # Reformat data to be compatible with SVM
    circles_data = circles_data.T
    print(circles_data.shape)
    circles_targets = circles_targets[:,0]
    print(circles_targets.shape)

    # Dvide the data for SVM
    train = int(0.8*N) 
    tune = train + int(0.1*N)
    test = tune + int(0.1*N)

    X_tr = circles_data[:train,:]
    X_tu = circles_data[train:tune,:]
    X_te = circles_data[tune:test,:]

    y_tr = circles_targets[:train]
    y_tu = circles_targets[train:tune]
    y_te = circles_targets[tune:test]

    # Fit the model with the training data
    clf = svm.SVC(kernel='rbf', C=100, gamma=0.1)
    clf.fit(X_tr, y_tr)
    # print(clf.predict([circles_data[0,:]])[0])
    # print(clf.predict([[2., 2.]])[0])
    
    # Report the success of the fit on the test data
    success = 0
    failure = 0
    counter = -1
    for target in range(len(y_te)):
        if clf.predict([X_te[counter,:]])[0] == y_te[counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} is: {}".format('Circles SVM', success / (success + failure)))

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

    N = divorce_targets.shape[0]
    # print(N)

    # # Do the things with the divorce data
    # P_C1 = 0.5

    # divorce_w = getWeights(divorce_data, divorce_targets, P_C1)
    # divorce_w0 = divorce_w[-1,0]
    # classify(divorce_data, divorce_targets, divorce_w, divorce_w0, 0.5, 'divorce data using gaussian generative')
    # plotROC(divorce_data, divorce_targets, divorce_w, divorce_w0, 'divorce data gaussian generative', 'b')
    
    # # # Use logistic regression with IRLS to recompute the weights. Feed ML estimate of the weights for the best initial condiditions.
    # divorce_w_irls = getWeightsIRLS(divorce_w, divorce_data, divorce_targets)
    # classify(divorce_data, divorce_targets, divorce_w_irls, divorce_w0, 0.5, 'divorce data using logistic regression')
    # plotROC(divorce_data, divorce_targets, divorce_w_irls, divorce_w0, 'divorce data logistic regression', 'g')

    # Reformat data to be compatible with SVM
    divorce_data = divorce_data.T
    print(divorce_data.shape)
    divorce_targets = divorce_targets[:,0]
    print(divorce_targets.shape)
    divorce_svm = svm.SVC(kernel='rbf', C=100, gamma=0.1)
    divorce_svm.fit(divorce_data, divorce_targets)
    
    # Find amount of successful classifications
    success = 0
    failure = 0
    counter = -1
    for target in np.nditer(divorce_targets):
        if divorce_svm.predict([divorce_data[counter,:]])[0] == divorce_targets[counter]:
            success += 1
        else:
            failure += 1
    print("The percentage of successfully classified {} is: {}".format('Divorce SVM', success / (success + failure)))

    # # plt.show()