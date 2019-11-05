import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt    

def basFunGauss(input_x, mu):
    # Using a gaussian basis function with mu's of random x value draws
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

def gaussKernel(x, x_prime):
    sig = 1
    k_of_x = (1 / sig*2*np.pi**(1/2)) * np.exp(np.linalg.norm(x - x_prime) / (2 * sig**2))
    return k_of_x

def plotModel(ax_number, N, title):
    # Plot original sine curve
    ax_number.plot(x_sin, y_sin, color='green')

    # Construct iota, the matrix of x values through basis functions, size N x M, where M = 9 in this case
    iota = np.array([[1, basFunGauss(x[0], x[0]), basFunGauss(x[0], x[1]), basFunGauss(x[0], x[2]), basFunGauss(x[0], x[3]), basFunGauss(x[0], x[4]), \
        basFunGauss(x[0], x[5]), basFunGauss(x[0], x[6]), basFunGauss(x[0], x[7]), basFunGauss(x[0], x[8])]])
    for i in range(1, N):
        add_me = np.array([[1, basFunGauss(x[i], x[0]), basFunGauss(x[i], x[1]), basFunGauss(x[i], x[2]), basFunGauss(x[i], x[3]), basFunGauss(x[i], x[4]), \
            basFunGauss(x[i], x[5]), basFunGauss(x[i], x[6]), basFunGauss(x[i], x[7]), basFunGauss(x[i], x[8])]])
        iota = np.concatenate((iota, add_me), axis=0) 

    # Create the target vector based on the data draw and use find the new mean of the weights
    target_vector = np.array([[target[0]]])
    for i in range(N-1):
        add_me = np.array([[target[i+1]]])
        target_vector = np.concatenate((target_vector, add_me), axis=0)

    # Construct the Gram matrix per Eq. 6.53
    K = (1/alpha) * np.matmul(iota, iota.T)
    # print(K.shape)

    # Construct the covariance matrix per Eq. 6.62
    delta = np.eye(N)
    C = K + (beta**(-1) * delta)
    # print(C.shape)

    # Construct a k vector
    k = np.zeros((N,1))
    for i in range(N):
        k[i, :] = gaussKernel(x[i], x[i+1])
    # print(k.shape)
    
    # Calculate c
    c = gaussKernel(x[N], x[N+1]) + (1/beta)

    m_next = np.matmul(np.matmul(k.T, np.linalg.inv(C)), target_vector) # Eq. 6.66
    # print(m_next.shape)
    covar_next = c - np.matmul(np.matmul(k.T, np.linalg.inv(C)), k) # Eq. 6.67
    # print(covar_next.shape)

    # # Update the covariance matrix
    S_N = np.linalg.inv(alpha*np.identity(10) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54

    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector) # Eq. 3.53
    print(m_N.shape)

    # Calculate the regression line for a given domain
    j = 100
    x_plot = np.linspace(-0.1, 1.1, j)
    mean_list = []
    low_list = []
    high_list = []
    for i in range(j):
        # phi of x, running through basis functions
        design_vector = np.transpose(np.array([[1, basFunGauss(x_plot[i], x[0]), basFunGauss(x_plot[i], x[1]), basFunGauss(x_plot[i], x[2]), \
            basFunGauss(x_plot[i], x[3]), basFunGauss(x_plot[i], x[4]), basFunGauss(x_plot[i], x[5]), \
            basFunGauss(x_plot[i], x[6]), basFunGauss(x_plot[i], x[7]), basFunGauss(x_plot[i], x[8])]]))
        mean_arr = np.matmul(np.transpose(m_N), design_vector)
        # Calculate the variance following Eq. 3.59
        variance_arr = 1/beta + np.matmul(np.matmul(np.transpose(design_vector), S_N), design_vector) # Eq. 
        low_arr = mean_arr - variance_arr**(1/2)
        high_arr = mean_arr + variance_arr**(1/2)
        mean_list.append(mean_arr[0,0])
        low_list.append(low_arr[0,0])
        high_list.append(high_arr[0,0])

    # # Plot the regression line with high and low variance bounding lines
    # ax_number.plot(x_plot, mean_list, color='red')
    # ax_number.fill_between(x_plot, low_list, high_list, color='mistyrose')
    # ax_number.set_xlabel('x')
    # ax_number.set_ylabel('t')
    # ax_number.set_title(title)

    # # Plot the draw of data points
    # for n in range(N):
    #     ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')

    return

if __name__ == "__main__":

    # Generate data points for the original sine curve
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(math.pi*2*x_sin)

    # Get a random draw of data
    N = 26
    x = np.random.random_sample(200,)
    # Ensure two of the data points occur at the limits
    x[3] = 0
    x[4] = 1
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = np.sin(math.pi*2*x) # find true value of target
    target += noise_t # Eq. 6.57 add gaussian noise to target

    # Construct the prior
    beta = (1/noise_sigma)**2
    alpha = 2.0
    m_0 = [0, 0]
    S_0 = alpha**(-1)*np.identity(2)

    # plot 2x2 subplots with the same x-axis and y-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.set_xlim(left = -0.1, right = 1.1)
    ax1.set_ylim(bottom = -1.2, top = 1.2)

    # Run the model with seperate draws of data
    plotModel(ax1, 1, "one data point")
    plotModel(ax2, 2, "two data points")
    plotModel(ax3, 4, "four data points")
    plotModel(ax4, 25, "25 data points")

    # plt.show()