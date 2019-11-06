import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp

def gaussKernel(input_x, mu):
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

def plotModel(ax_number, N, title):
    # Plot original sine curve
    ax_number.plot(x_sin, y_sin, color='green')

    # Create the target vector with shape based on the data draw and use find the new mean of the weights
    target_vector = np.array([[target[0]]])
    for i in range(N-1):
        add_me = np.array([[target[i+1]]])
        target_vector = np.concatenate((target_vector, add_me), axis=0)

    # Construct the gram matrix per Eq. 6.54
    K = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            K[n,m] = gaussKernel(x[n], x[m])

    # Construct the covariance matrix per Eq. 6.62
    delta = np.eye(N)
    C = K + ((1/beta) * delta)
    C_inv = np.linalg.inv(C)

    # Find mean for each new x value in the linspace using a gaussian process
    N_plot = 100
    x_list = np.linspace(-0.1, 1.1, N_plot)
    c = np.zeros((1,1))
    mean_list = []
    mean_low = []
    mean_high = []
    for i in range(len(x_list)):
        k = np.zeros((N, 1))
        for j in range(N):
            k[j, :] = gaussKernel(x[j], x_list[i])
        m_next = np.matmul(k.T, C_inv)
        m_next = np.matmul(m_next, target_vector) # Eq. 6.66
        mean_list.append(m_next[0,0])

        c[0,0] = gaussKernel(x_list[i], x_list[i]) + (1/beta)
        covar_next = np.matmul(k.T, C_inv) 
        covar_next = c - np.matmul(covar_next, k) # Eq. 6.67
        
        # Find predicition accuracy by adding/subtracting covariance to/from mean
        mean_low.append(m_next[0,0] - np.sqrt(covar_next[0,0]))
        mean_high.append(m_next[0,0] + np.sqrt(covar_next[0,0]))

    # Generate gaussian sinusoid guess based generated means
    ax_number.plot(x_list, mean_list, color = 'r')
    ax_number.fill_between(x_list, mean_low, mean_high, color='mistyrose')
    ax_number.set_xlabel('x')
    ax_number.set_ylabel('t')
    ax_number.set_title(title)

    # Plot the particular draw of data points
    for n in range(N):
        ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')
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

    # Define statistical parameters
    beta = (1/noise_sigma)**2
    alpha = 2.0

    # plot 2x2 subplots with the same x-axis and y-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    ax1.set_xlim(left = -0.1, right = 1.1)
    ax1.set_ylim(bottom = -1.2, top = 1.2)

    # Run the model with seperate draws of data
    plotModel(ax1, 1, "one data point")
    plotModel(ax2, 2, "two data points")
    plotModel(ax3, 4, "four data points")
    plotModel(ax4, 25, "25 data points")

    plt.show()