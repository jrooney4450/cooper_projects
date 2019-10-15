import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt    

def basFunGauss(input_x, mu):
    # Using a gaussian basis function with mu's of random x value draws
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

def plotModel(ax_number, N):
    # Plot original sine curve
    ax_number.plot(x_sin, y_sin, color='green')

    # Matrix of features through basis functions, size NxM, where M=9 in this case for 9 weights
    iota = np.array([[1, basFunGauss(x[0], x[0]), basFunGauss(x[0], x[1]), basFunGauss(x[0], x[2]), basFunGauss(x[0], x[3]), basFunGauss(x[0], x[4]), \
        basFunGauss(x[0], x[5]), basFunGauss(x[0], x[6]), basFunGauss(x[0], x[7]), basFunGauss(x[0], x[8])]])

    for i in range(1, N):
        add_me = np.array([[1, basFunGauss(x[i], x[0]), basFunGauss(x[i], x[1]), basFunGauss(x[i], x[2]), basFunGauss(x[i], x[3]), basFunGauss(x[i], x[4]), \
            basFunGauss(x[i], x[5]), basFunGauss(x[i], x[6]), basFunGauss(x[i], x[7]), basFunGauss(x[i], x[8])]])
        iota = np.concatenate((iota, add_me), axis=0) 
    # print("iota is: {}".format(iota))

    # Inverse of updated covariance matrix
    S_N = np.linalg.inv(alpha*np.identity(10) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54
    # print("covariance matrix is: {}".format(S_N))

    # Create the target vector and find the new mean of the weights
    target_vector = np.array([[target[0]]])
    for i in range(N-1):
        add_me = np.array([[target[i+1]]])
        target_vector = np.concatenate((target_vector, add_me), axis=0)
    # print("target vector is: {}".format(target_vector))
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector) # Eq. 3.53
    # print("mean vector {}".format(m_N))

    # # find variance per Eq. 3.59
    # for i in m_N.shape:
    #     phi_of_x_array = np.array([[1, basFunGauss(x[5], 0), basFunGauss(x[5], 1), basFunGauss(x[5], 2), basFunGauss(x[5], 3), basFunGauss(x[5], 4), \
    #         basFunGauss(x[5], 5), basFunGauss(x[5], 6), basFunGauss(x[5], 7), basFunGauss(x[5], 8)]])
    #     phi_of_x_array = np.transpose(phi_of_x_array)
    # # print("phi of x: {}".format(phi_of_x_array)) # grab a column of iota

    # iota_T = np.transpose(iota)
    # print(iota_T[:,0])
    # phi_of_x_array = iota_T[:,0]

    # variance_N = 1/beta + np.matmul((np.transpose(phi_of_x_array), S_N), phi_of_x_array)
    # variance_N = np.matmul(np.transpose(phi_of_x_array), S_N)
    # variance_N = np.matmul(variance_N, phi_of_x_array) + 1/beta
    # print("the varaince is: {}".format(variance_N))

    # Plot the regression line with the calculated weights
    j = 100
    x_plot = np.linspace(0, 1, j)
    mean_j = []
    for i in range(j):
        design_vector = np.transpose(np.array([[1, basFunGauss(x_plot[i], x[0]), basFunGauss(x_plot[i], x[1]), basFunGauss(x_plot[i], x[2]), \
            basFunGauss(x_plot[i], x[3]), basFunGauss(x_plot[i], x[4]), basFunGauss(x_plot[i], x[5]), \
            basFunGauss(x_plot[i], x[6]), basFunGauss(x_plot[i], x[7]), basFunGauss(x_plot[i], x[8])]]))
        # print(design_vector)
        result = np.matmul(np.transpose(m_N), design_vector)
        mean_j.append(result[0,0])
    
    # mean_j = np.ndarray.tolist(mean_j)
    # print(x_plot.shape)
    # print(mean_j.shape)

    ax_number.plot(x_plot, mean_j, color='red')

    # ax_number.plot(x_plot, m_N[0] + m_N[1]*x_plot + m_N[2]*x_plot**2 + m_N[3]*x_plot**3 + m_N[4]*x_plot**4 + m_N[5]*x_plot**5 + \
    #     m_N[6]*x_plot**6 + m_N[7]*x_plot**7 + m_N[8]*x_plot**8 + m_N[9]*x_plot**9, color = 'red')

    # Plot the data points
    for n in range(N):
        ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')

if __name__ == "__main__":

    # Generate points for the original sine curve
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(math.pi*2*x_sin)

    # Need random data set of max N = 25
    N = 25
    x = np.random.random_sample(200,)
    # ensure two of the data points occur at the limits
    x[3] = 0
    x[4] = 1
    noise_sigma = 0.2
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

    # print(basFunGauss(0.9, 3))

    plotModel(ax1, 1)
    plotModel(ax2, 2)
    plotModel(ax3, 4)
    plotModel(ax4, 25)

    plt.show()