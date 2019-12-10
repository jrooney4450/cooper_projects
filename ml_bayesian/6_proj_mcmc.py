import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sklearn.gaussian_process as gp

def basFunGauss(input_x, mu):
    # Using a gaussian basis function with mu's of random x value draws
    sig = 0.2
    phi_of_x = (1 / sig*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*sig**2)))
    return phi_of_x

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

    # Update the covariance matrix
    S_N = np.linalg.inv(alpha*np.identity(10) + beta*(np.matmul(np.transpose(iota),iota))) # Eq. 3.54

    # Create the y vector based on the data draw and use find the new mean of the weights
    target_vector = np.array([[y[0]]])
    for i in range(N-1):
        add_me = np.array([[y[i+1]]])
        target_vector = np.concatenate((target_vector, add_me), axis=0)
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), target_vector) # Eq. 3.53
    print(m_N)

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
        variance_arr = 1/beta + np.matmul(np.matmul(np.transpose(design_vector), S_N), design_vector)
        low_arr = mean_arr - variance_arr**(1/2)
        high_arr = mean_arr + variance_arr**(1/2)
        mean_list.append(mean_arr[0,0])
        low_list.append(low_arr[0,0])
        high_list.append(high_arr[0,0])

    # Plot the regression line with high and low variance bounding lines
    ax_number.plot(x_plot, mean_list, color='red')
    # ax_number.plot(x_plot, mean_arr, color='red')
    ax_number.fill_between(x_plot, low_list, high_list, color='mistyrose')
    # ax_number.fill_between(x_plot, low_arr, high_arr, color='mistyrose')
    ax_number.set_xlabel('x')
    ax_number.set_ylabel('t')
    ax_number.set_title(title)

    # Plot the draw of data points
    for n in range(N):
        ax_number.scatter(x[n], y[n], facecolors='none', edgecolors='blue')

    return

def plotMCMCModel(x, t, N):
    x = x[0:N]
    t = t[0:N]

    M = 4 # Number of weights
    w = (np.random.random_sample(M) * 2) - 1 # random guess for the weights

    # parameters
    sig = 0.2
    beta = (1/sig)**2
    alpha = 2.0

    # construt a prior
    m0 = np.zeros((M,))
    S0 = np.identity(M)

    S = 0.04*S0 # Const cov for mean refresh

    z_prev = w
    z_star = np.random.multivariate_normal(z_prev, S) # gen first sample
    z = [z_prev, z_star] # store into list
    p = [0, 0] # probabilities of z_prev and z_star
    log_prior = multivariate_normal.pdf(z[0], m0, S0)

    for _ in range(10000):
        for k in range(2):
            like_sum = 0

            for i in range(N):
                phi = np.array([1, basFunGauss(x[i], x[0]), basFunGauss(x[i], x[1]), basFunGauss(x[i], x[2])])
                like = np.abs(mlab.normpdf(t[i], np.dot(z[k], phi), beta**(-1))) # Eq. 3.10 for likelihood
                like_sum += like
            
            log_like = np.log(like_sum)
            log_prior = np.log(np.abs(multivariate_normal.pdf(z[k], m0, S)))
            p[k] = log_prior + log_like # this is the posterior

        u = np.random.uniform(0, 1)
        A = min([1.0, p[0] / p[1]]) # Eq. 11.33 - Metropolis criterion
        if A > u: # accept this point
            print('point accepted')
            z[0] = z[1] # z_star assigned to z_prev
            z[1] = np.random.multivariate_normal(z[0], S) # draw another data point
            # log_prior = p[1] # set equal to previous posterior
        else: # reject this point 
            print('point rejected')
            z[1] = np.random.multivariate_normal(z[0], S) # try again with another data point


    m_N = z[0].reshape(-1, 1)
    print('weights from mcmc: {}'.format(m_N))

    S_N = S

    # Calculate the regression line for a given domain
    I = 100
    x_plot = np.linspace(-0.1, 1.1, I)
    mean_list = []
    low_list = []
    high_list = []
    for i in range(I):
        # phi of x, running through basis functions
        design_vector = np.transpose(np.array([1, basFunGauss(x_plot[i], x[0]), basFunGauss(x_plot[i], x[1]), basFunGauss(x_plot[i], x[2])]))
        mean_arr = np.matmul(np.transpose(m_N), design_vector)
        # Calculate the variance following Eq. 3.59
        variance_arr = 1/beta + np.matmul(np.matmul(np.transpose(design_vector), S_N), design_vector)
        low_arr = mean_arr - variance_arr**(1/2)
        high_arr = mean_arr + variance_arr**(1/2)
        mean_list.append(mean_arr[0])
        low_list.append(low_arr[0])
        high_list.append(high_arr[0])
        # Plot the regression line with high and low variance bounding lines
    
    fig, ax2 = plt.subplots()
    ax2.plot(x_plot, mean_list, color='red')
    ax2.fill_between(x_plot, low_list, high_list, color='mistyrose')

    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('MCMC Linear Regression')

    # Plot the draw of data points
    for n in range(N):
        ax2.scatter(x[n], t[n], facecolors='none', edgecolors='blue')

##############################################################################################

    # # Do regression the previous way for comparison
    # iota = np.array([[1, basFunGauss(x[0], x[0]), basFunGauss(x[0], x[1]), basFunGauss(x[0], x[2])]])
    # for i in range(1, N):
    #     add_me = np.array([[1, basFunGauss(x[i], x[0]), basFunGauss(x[i], x[1]), basFunGauss(x[i], x[2])]])
    #     iota = np.concatenate((iota, add_me), axis=0) 

    # # # Update the covariance matrix
    # t = t.reshape(-1, 1)
    # S_N = np.linalg.inv(alpha*np.identity(M) + beta*(np.matmul(np.transpose(iota), iota))) # Eq. 3.54
    # m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), t) # Eq. 3.53
    # print('Old weights are: {}'.format(m_N))

    # # Calculate the regression line for a given domain
    # mean_list = []
    # low_list = []
    # high_list = []
    # for i in range(I):
    #     # phi of x, running through basis functions
    #     design_vector = np.transpose(np.array([1, basFunGauss(x_plot[i], x[0]), basFunGauss(x_plot[i], x[1]), basFunGauss(x_plot[i], x[2])]))
    #     mean_arr = np.matmul(np.transpose(m_N), design_vector)
    #     # Calculate the variance following Eq. 3.59
    #     variance_arr = 1/beta + np.matmul(np.matmul(np.transpose(design_vector), S_N), design_vector)
    #     low_arr = mean_arr - variance_arr**(1/2)
    #     high_arr = mean_arr + variance_arr**(1/2)
    #     mean_list.append(mean_arr[0])
    #     low_list.append(low_arr[0])
    #     high_list.append(high_arr[0])
    #     # Plot the regression line with high and low variance bounding lines
    
    # fig, ax_number = plt.subplots()
    # ax_number.plot(x_plot, mean_list, color='red')
    # ax_number.fill_between(x_plot, low_list, high_list, color='mistyrose')

    # ax_number.set_xlabel('x')
    # ax_number.set_ylabel('t')
    # ax_number.set_title('bayesian linear regression')
    
    # # Plot the draw of data points
    # for n in range(N):
    #     ax_number.scatter(x[n], t[n], facecolors='none', edgecolors='blue')
            

def main():
    # Generate data points for the original sine curve
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(math.pi*2*x_sin)

    # Get a random draw of data
    x = np.random.random_sample(200,)
    # Ensure two of the data points occur at the limits
    x[3] = 0
    x[4] = 1
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    t = np.sin(math.pi*2*x) # find true value of y
    t += noise_t # Eq. 3.7 add gaussian noise to y

    # Construct the prior
    beta = (1/noise_sigma)**2
    alpha = 2.0
    m_0 = [0, 0]
    S_0 = alpha**(-1)*np.identity(2)

    # # plot 2x2 subplots with the same x-axis and y-axis
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    # ax1.set_xlim(left = -0.1, right = 1.1)
    # ax1.set_ylim(bottom = -1.2, top = 1.2)

    # # Run the model with seperate draws of data
    # plotModel(ax1, 1, "one data point")
    # plotModel(ax2, 2, "two data points")
    # plotModel(ax3, 4, "four data points")
    # plotModel(ax4, 25, "25 data points")

    # plt.figure(2)
    N = 25
    plotMCMCModel(x, t, N)

    plt.show()

if __name__ == "__main__":
    main()