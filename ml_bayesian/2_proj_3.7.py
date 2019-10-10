import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plotMultivariate(mu_vector, covariance_mat):
    x_graphing = np.linspace(-1,1,500)
    y_graphing = np.linspace(-1,1,500)
    X, Y = np.meshgrid(x_graphing,y_graphing)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    rv = multivariate_normal(mu_vector, covariance_mat)
    plt.figure()
    plt.contourf(X, Y, rv.pdf(pos))
    # plt.scatter(w0_true, w1_true) #TODO add true point
    plt.xlabel('w0')
    plt.ylabel('w1')

    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    # plt.set_ylabel('w1')
    # ax.set_zlabel('Z axis')
    # ax.set_top_view()
    # ax.tick_params(labelrotation=45)
    # ax.view_init(elev=90., azim=270)

# # Draw lines using w0 and w1 from prior TODO add random line plotter follow covariance matrix
# p_N = 6 # number of samples
# w0 = np.random.normal(w, w0_sigma, p_N)
# w1 = np.random.normal(w1_mu, w1_sigma, p_N)

# fig = plt.figure()
# ax = plt.axes()

# for n in range(w0.size):
#     ax.plot(x, w0[n] + w1[n]*x)

# plt.title("data space")
# plt.xlabel("x")
# plt.ylabel("y")

#### Attempt with first data point

# Use gaussian basis functions Eq. 3.4
# For much of the book the vector phi(x) = x

def plotPosterior(alpha, beta, N_target):
    # Matrix of features through basis functions, size Nx2
    iota = np.array([[1, x[0]]])
    if N_target > 1:
        for i in range(N_target-1):
            add_me = np.array([[1, x[i+1]]]) # using the identity for each basis function
            iota = np.concatenate((iota, add_me), axis=0)

    # Inverve of updated covariance matrix
    S_N_inverse = (alpha*np.identity(2) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54
    # Updated covariance matrix of posterior 
    S_N = np.linalg.inv(S_N_inverse) # updated variance

    # Create the target vector
    target_vector = np.array([[target[0]]])
    if N_target > 1:
        for i in range(N_target-1):
            add_me = np.array([[target[i+1]]])
            target_vector = np.concatenate((target_vector, add_me), axis=0)

    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector)
    
    # Rearrange the mean vector such that the plotter can handle
    m_N = [m_N.item(0), m_N.item(1)]

    plotMultivariate(m_N, S_N)
    return

if __name__ == "__main__":

    # True weights we wish to recover
    a0 = -0.3
    a1 = 0.5

    # Generate data points from weights
    N = 20
    x = 2 * np.random.random_sample(N,) - 1

    # Add noise to the data
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = a0 + a1*x # find true value of target
    target += noise_t # Eq. 3.7 add gaussian noise to target

    # Plot the prior
    beta = (1/noise_sigma)**2
    alpha = 2.0
    m_0 = [0, 0]
    S_0 = alpha**(-1)*np.identity(2)

    plotMultivariate(m_0, S_0) # plot prior
    
    plotPosterior(alpha, beta, 1)
    plotPosterior(alpha, beta, 2)
    plotPosterior(alpha, beta, 20)

    plt.show()