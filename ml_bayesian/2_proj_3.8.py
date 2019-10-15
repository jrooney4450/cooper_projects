import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt    

def basFunGauss(input_x, j):
    # Using a gaussian basis function
    # scale_all = 1
    # mu = [scale_all, scale_all, scale_all, scale_all, scale_all, scale_all, scale_all, scale_all, scale_all]
    mu = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    s = 1
    phi_of_x = np.exp((-(input_x-mu[j])**2/(2*s**2)))
    return phi_of_x

# def genIota(M, N):
#     iota_row = np.array([1])
#     iota = np.zeros(1, M)
#     # fill the first row
#     for row in range(N):
#         for column in range(M):
#             print(column)
#             add_me = np.array([basFunGauss(x[row], column)])
#             iota = np.concatenate((iota_row, add_me), axis = 0)
#         iota = np.concatenate((iota, , axis=0))
#     print(iota)
#     return

def plotModel(ax_number, N):
    # Plot original sine curve
    ax_number.plot(x_sin, y_sin, color='green')

    # Matrix of features through basis functions, size NxM, where M=9 in this case for 9 weights
    iota = np.array([[1, basFunGauss(x[0], 0), basFunGauss(x[0], 1), basFunGauss(x[0], 2), basFunGauss(x[0], 3), basFunGauss(x[0], 4), basFunGauss(x[0], 5), basFunGauss(x[0], 6), basFunGauss(x[0], 7), basFunGauss(x[0], 8)]])
    if N > 1:
        for i in range(1, N):
            add_me = np.array([[1, basFunGauss(x[i], 0), basFunGauss(x[i], 1), basFunGauss(x[i], 2), basFunGauss(x[i], 3), basFunGauss(x[i], 4), \
                basFunGauss(x[i], 5), basFunGauss(x[i], 6), basFunGauss(x[i], 7), basFunGauss(x[i], 8)]])
            iota = np.concatenate((iota, add_me), axis=0)

    # Inverse of updated covariance matrix
    S_N_inverse = (alpha*np.identity(10) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54
    # Updated covariance matrix of posterior
    S_N = np.linalg.inv(S_N_inverse)

    # Create the target vector and find the new mean of the weights
    target_vector = np.array([[target[0]]])
    if N > 1:
        for i in range(N-1):
            add_me = np.array([[target[i+1]]])
            target_vector = np.concatenate((target_vector, add_me), axis=0)
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector) # Eq. 3.53

    # Plot the regression line with the calculated weights
    x_plot = np.linspace(-0.1, 1.1, 100)
    ax_number.plot(x_plot, m_N[0] + m_N[1]*x_plot + m_N[2]*x_plot**2 + m_N[3]*x_plot**3 + m_N[4]*x_plot**4 + m_N[5]*x_plot**5 + m_N[6]*x_plot**6 + m_N[7]*x_plot**7 + m_N[8]*x_plot**8 + m_N[9]*x_plot**9, color = 'red')

    print(m_N)

    # Plot the data points
    for n in range(N):
        ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')

if __name__ == "__main__":

    # Generate points for the original sine curve
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(math.pi*2*x_sin)

    # Need random data set of max N = 25
    N = 25
    x = np.random.random_sample(N,)
    noise_sigma = 0.1
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = np.sin(math.pi*2*x) # find true value of target
    target += noise_t # Eq. 3.7 add gaussian noise to target

    beta = (1/noise_sigma)**2
    alpha = 2.0
    m_0 = [0, 0]
    S_0 = alpha**(-1)*np.identity(2)

    # plot 2x2 subplots with the same x-axis and y-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.set_xlim(left = -0.1, right = 1.1)
    ax1.set_ylim(bottom = -1.2, top = 1.2)

    # print(genIota(9, 25))

    plotModel(ax1, 1)
    plotModel(ax2, 2)
    plotModel(ax3, 4)
    plotModel(ax4, 25)

    plt.show()