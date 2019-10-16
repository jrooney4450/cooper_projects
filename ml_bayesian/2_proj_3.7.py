import math
import numpy as np
from scipy.stats import multivariate_normal,norm
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def plotLikelihood(ax_number, N_target, target_vector, m_N, S_N):
    # Initialize paramemeters required to search for likeeihood distribution
    j = 100
    w0 = np.linspace(-1.1, 1.1, j)
    w1 = np.linspace(-1.1, 1.1, j)
    W0, W1 = np.meshgrid(w0, w1)
    probability = np.zeros((j, j))

    # Find the probability that a target value is near a mean given by the combination of all possible weights through the line equation. This is the likelihood.
    for i in range(N_target):
        for j in range(j):
            for k in range(j):
                probability[k, j] = mlab.normpdf(target_vector.item(i), w0[j] + x[i] * w1[k], noise_sigma)

    # Plot probability distribution as a countour
    ### NOTE! The countour gets cut off along the diagonal. Not sure why this happens but could not figure
    # out how to debug. The theory behind the line and the shape of it matches Figure 3.7.
    ax_number.contourf(W0, W1, probability)
    ax_number.scatter(a[0], a[1], color='white', marker='+') # plot true point
    ax_number.set_xlabel('w0')
    ax_number.set_ylabel('w1')

def plotLines(ax_number, N_target, m_N, S_N):
    # Generate random weights with updated mean and covariance to plot 6 lines
    p_N = 6 # number of samples
    w0 = np.random.normal(m_N[0], S_N[0, 0], p_N)
    w1 = np.random.normal(m_N[1], S_N[1, 1], p_N)

    # Plot 6 lines given the distribution of the weights
    for n in range(w0.size):
        ax_number.plot(x, w0[n] + w1[n]*x, color='red')

    # Plot the data points with noise
    for n in range(N_target):
        ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')
    ax_number.set_xlabel('x')
    ax_number.set_ylabel('y')

def plotMultivariate(ax_number, mu_vector, covariance_mat):
    # Initialized parameters needed for plotting
    x_graphing = np.linspace(-1,1,100)
    y_graphing = np.linspace(-1,1,100)
    X, Y = np.meshgrid(x_graphing, y_graphing)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    # Input the found mean and covariance values
    rv = multivariate_normal(mu_vector, covariance_mat)

    # Find the PDF throughout the grid given those means and covarainces
    ax_number.contourf(X, Y, rv.pdf(pos)) # Create contour plot at posiitons within PDF
    ax_number.scatter(a[0], a[1], color='white', marker='+') # Plot true point
    ax_number.set_xlabel('w0')
    ax_number.set_ylabel('w1')

def plotPosterior(ax_number1, ax_number2, ax_number3, N_target):
    # Matrix of features through basis functions, size NxM, where M=2 in this case
    iota = np.array([[1, x[0]]])
    if N_target > 1:
        for i in range(N_target-1):
            add_me = np.array([[1, x[i+1]]]) # using the identity for each basis function
            iota = np.concatenate((iota, add_me), axis=0)

    # Updated covariance matrix of posterior
    S_N = np.linalg.inv(alpha*np.identity(2) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54

    # Create the target vector and find the new mean of the weights
    target_vector = np.array([[target[0]]])
    if N_target > 1:
        for i in range(N_target-1):
            add_me = np.array([[target[i+1]]])
            target_vector = np.concatenate((target_vector, add_me), axis=0)
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector) # Mean vector Eq. 3.53
    
    # Rearrange the mean vector such that the plotter can handle
    m_N = [m_N.item(0), m_N.item(1)]

    # Find the likelihood of the target given the data draw
    plotLikelihood(ax_number1, N_target, target_vector, m_N, S_N)

    # Plot the probability distribution given the found mean and covariance
    plotMultivariate(ax_number2, m_N, S_N)

    # Plot 6 randomly generated lines given the found mean and covariance of the weights
    plotLines(ax_number3, N_target, m_N, S_N)
    return

if __name__ == "__main__":

    # True weights we wish to recover
    a = [0.5, -0.5]

    # Generate true data points from true weights
    N = 20
    x = 2 * np.random.random_sample(N,) - 1

    # Add gaussian noise to the true data points
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = a[0] + a[1]*x # find true value of target
    target += noise_t # Eq. 3.7 add gaussian noise to target

    # Generate a prior with a mean of 0 and diagonal covariance
    beta = (1/noise_sigma)**2
    alpha = 2.0
    m_0 = [0, 0]
    S_0 = alpha**(-1)*np.identity(2)

    # Initialize 4x3 subplots with the same x-axis and y-axis scaling
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, sharex=True, sharey=True)
    ax1.set_xlim(left = -1, right = 1)
    ax1.set_ylim(bottom = -1, top = 1)

    # Plot the prior and six lines from the probability of the weights
    plotMultivariate(ax2, m_0, S_0)
    plotLines(ax3, 0, m_0, S_0)
    
    # Improve the model by using draws from the data and plot needed figures
    plotPosterior(ax4, ax5, ax6, 1) # 1 data point
    plotPosterior(ax7, ax8, ax9, 2) # 2 data points
    plotPosterior(ax10, ax11, ax12, 20) # 20 data points

    plt.show()