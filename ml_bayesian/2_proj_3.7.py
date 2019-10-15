import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plotLines(ax_number, N_target, m_N, S_N):
    # Generate random weights with updated mean and covariance to plot 6 lines
    p_N = 6 # number of samples
    w0 = np.random.normal(m_N[0], S_N[0, 0], p_N)
    w1 = np.random.normal(m_N[1], S_N[1, 1], p_N)

    # Plot 6 lines given the distribution of the weights
    for n in range(w0.size):
        ax_number.plot(x, w0[n] + w1[n]*x, color='red')

    # Plot the data points
    for n in range(N_target):
        ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')
    
    ax_number.set_xlabel('x')
    ax_number.set_ylabel('y')

    return

def plotMultivariate(ax_number, mu_vector, covariance_mat):
    x_graphing = np.linspace(-1,1,500)
    y_graphing = np.linspace(-1,1,500)
    X, Y = np.meshgrid(x_graphing,y_graphing)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(mu_vector, covariance_mat)

    ax_number.contourf(X, Y, rv.pdf(pos))
    ax_number.scatter(a[0], a[1], color='white', marker='+')
    ax_number.set_xlabel('w0')
    ax_number.set_ylabel('w1')

# def plotLikelihood()

def plotPosterior(ax_number1, ax_number2, ax_number3, N_target):
    # Matrix of features through basis functions, size NxM, where M=2 in this case
    iota = np.array([[1, x[0]]])
    if N_target > 1:
        for i in range(N_target-1):
            add_me = np.array([[1, x[i+1]]]) # using the identity for each basis function
            iota = np.concatenate((iota, add_me), axis=0)
    print(iota)

    # Inverve of updated covariance matrix
    S_N_inverse = (alpha*np.identity(2) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54
    # Updated covariance matrix of posterior
    S_N = np.linalg.inv(S_N_inverse)

    # Create the target vector and find the new mean of the weights
    target_vector = np.array([[target[0]]])
    if N_target > 1:
        for i in range(N_target-1):
            add_me = np.array([[target[i+1]]])
            target_vector = np.concatenate((target_vector, add_me), axis=0)
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector) # Eq. 3.53
    
    # Rearrange the mean vector such that the plotter can handle
    m_N = [m_N.item(0), m_N.item(1)]

    plotMultivariate(ax_number2, m_N, S_N)

    # TODO determine likelihood and plot
    weights_ml = np.matmul(np.transpose(iota), iota)
    weights_ml = np.linalg.inv(weights_ml)
    weights_ml = np.matmul(weights_ml, np.transpose(iota))
    weights_ml = np.matmul(weights_ml, target_vector)
    weights_ml = [weights_ml.item(0), weights_ml.item(1)]
    print(weights_ml)

    x_like = np.linspace(-1,1,100)
    ax_number1.plot(x_like, weights_ml[0] + weights_ml[1]*x_like)

    # plotMultivariate(ax_number1, weights_ml, S_0)

    plotLines(ax_number3, N_target, m_N, S_N)
    return

if __name__ == "__main__":

    # True weights we wish to recover
    a = [-0.3, 0.5]

    # Generate true data points from weights
    N = 20
    x = 2 * np.random.random_sample(N,) - 1

    # Add gaussian noise to the true data points
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = a[0] + a[1]*x # find true value of target
    target += noise_t # Eq. 3.7 add gaussian noise to target

    # Plot the prior with values give by Bishop
    beta = (1/noise_sigma)**2
    alpha = 2.0
    m_0 = [0, 0]
    S_0 = alpha**(-1)*np.identity(2)

    # plot 4x3 subplots with the same x-axis and y-axis scaling
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, sharex=True, sharey=True)
    ax1.set_xlim(left = -1, right = 1)
    ax1.set_ylim(bottom = -1, top = 1)

    # # Plot the prior and six lines from the estimated weights
    plotMultivariate(ax2, m_0, S_0)
    plotLines(ax3, 0, m_0, S_0)
    
    # Improve the model by using draws from the data
    plotPosterior(ax4, ax5, ax6, 1) # 1 data point
    plotPosterior(ax7, ax8, ax9, 2) # 2 data points
    plotPosterior(ax10, ax11, ax12, 20) # 20 data points

    plt.show()