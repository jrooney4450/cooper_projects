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

def getPhi(M, x1, x2, i):
    phi = np.zeros(M,)
    phi[0] = 1
    for m in range(M - 1):
        phi[m + 1] = basFunGauss(x1[i], x2[m])
    return phi


def plotMCMCModel(x, t, N, x_sin, y_sin):
    x = x[0:N]
    t = t[0:N]

    M = 2 # Number of weights
    w = (np.random.random_sample(M) * 2) - 1 # random guess for the weights

    # parameters
    sig = 0.2
    beta = (1/sig)**2
    alpha = 2.0

    # construt a prior
    m0 = np.zeros((M,))
    k = 1
    S0 = k * np.identity(M)
    
    iota = np.zeros((N, M))
    for i in range(N):
        iota[i, :] = getPhi(M, x, x, i)

    # # Update the covariance matrix
    t = t.reshape(-1, 1)
    S_N = np.linalg.inv(alpha*np.identity(M) + beta*(np.matmul(np.transpose(iota), iota))) # Eq. 3.54
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), t) # Eq. 3.53


##############################################################################################

    S = 0.04*S0 # Const cov for mean refresh

    z_prev = w
    z_star = np.random.multivariate_normal(z_prev, S) # gen first sample
    z = [z_prev, z_star] # store into list
    p = [0, 0] # probabilities of z_prev and z_star
    log_prior = multivariate_normal.pdf(z[0], m0, S0)

    # Make a movie! Plot dataset with ellipse for newly calculated parameters
    fig, ax_loop = plt.subplots()
    ax_loop.set_title('2D Expectation Maximization')
    ax_loop.set_xlabel('w0')
    ax_loop.set_ylabel('w1') 
    ax_loop.set_xlim(-5, 5)
    ax_loop.set_ylim(-5, 5)

    log_prior = np.log(multivariate_normal.pdf(z[0], m0, S0))

    min_like = 0.0 # store point with maximum likelihood
    min_like_z = np.zeros((2,))

    for _ in range(10000):
        for k in range(2):
            like_sum = 0

            for i in range(N):
                # phi = np.array([1, basFunGauss(x[i], x[0]), basFunGauss(x[i], x[1]), basFunGauss(x[i], x[2])]) # M = 4
                # phi = np.array([1, basFunGauss(x[i], x[0])])
                phi = getPhi(M, x, x, i)
                like = mlab.normpdf(t[i], np.dot(z[k], phi), beta**(-1)) # Eq. 3.10 for likelihood
                like_sum += like
            
            log_like = np.log(like_sum)
            # print('{} log likelihood'.format(log_like))
            log_prior = np.log(multivariate_normal.pdf(z[k], m0, S0))
            p[k] = log_prior + log_like # this is the posterior

            if np.abs(log_like) < min_like:
                min_like_z = z[k]

        u = np.random.uniform(0, 1)
        A = min([1.0, p[0] / p[1]]) # Eq. 11.33 - Metropolis criterion

        print('{} log likelihood'.format(log_like))
        print('{}: old weights'.format(m_N[:,0]))
        print('{}: best weight'.format(min_like_z))
        print('{}: new weights\n'.format(z[0]))

        if A > u: # accept this point
            # print('point accepted')
            ax_loop.scatter(z[0][0], z[0][1], color = 'red')
            ax_loop.scatter(z[1][0], z[1][1], color = 'green', s=0.8) # plot z_star as success
            z[0] = z[1] # z_star assigned to z_prev
            z[1] = np.random.multivariate_normal(z[0], S) # draw another data point
            # log_prior = p[1] # set equal to previous posterior

        else: # reject this point 
            # print('point rejected')
            ax_loop.scatter(z[1][0], z[1][1], color = 'red', s=0.8) # plot z_star as failure
            z[1] = np.random.multivariate_normal(z[0], S) # try again with another data point

        # # Make a movie
        # plt.ion()
        # plt.show(block=False)
        # plt.pause(0.1)
        # plt.close()

    ax_loop.scatter(z[0][0], z[0][1], color = 'cyan', marker='x', s=2)
    ax_loop.scatter(min_like_z[0], min_like_z[1], color = 'black', marker='x', s=2)

    # Chose which value to use from MCMC
    print('last weight from mcmc: {}'.format(z[0]))
    print('min like weight from mcmc: {}'.format(min_like_z))
    # best_z = z[0].reshape(-1, 1)
    best_z = min_like_z.reshape(-1, 1)

    # Calculate the regression line for a given domain
    I = 100
    x_plot = np.linspace(-0.1, 1.1, I)
    mean_list = []
    low_list = []
    high_list = []
    for i in range(I):
        # phi of x, running through basis functions
        design_vector = getPhi(M, x_plot, x, i).reshape(-1, 1)
        mean_arr = np.matmul(np.transpose(best_z), design_vector)
        # Calculate the variance following Eq. 3.59
        variance_arr = 1/beta + np.matmul(np.matmul(np.transpose(design_vector), S), design_vector)
        low_arr = mean_arr - variance_arr**(1/2)
        high_arr = mean_arr + variance_arr**(1/2)
        mean_list.append(mean_arr[0, 0])
        low_list.append(low_arr[0, 0])
        high_list.append(high_arr[0, 0])
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

#############################################################################################

    # Calculate the regression line for a given domain
    I = 100
    x_plot = np.linspace(-0.1, 1.1, I)
    mean_list = []
    low_list = []
    high_list = []

    for i in range(I):
        design_vector = getPhi(M, x_plot, x, i).reshape(-1, 1)
        mean_arr = np.matmul(np.transpose(m_N), design_vector)

        # Calculate the variance following Eq. 3.59
        variance_arr = 1/beta + np.matmul(np.matmul(np.transpose(design_vector), S_N), design_vector)
        low_arr = mean_arr - variance_arr**(1/2)
        high_arr = mean_arr + variance_arr**(1/2)
        mean_list.append(mean_arr[0,0])
        low_list.append(low_arr[0,0])
        high_list.append(high_arr[0,0])
        
    # Plot the regression line with high and low variance bounding lines
    fig, ax_number = plt.subplots()
    ax_number.plot(x_plot, mean_list, color='red')
    ax_number.fill_between(x_plot, low_list, high_list, color='mistyrose')

    ax_number.set_xlabel('x')
    ax_number.set_ylabel('t')
    ax_number.set_title('bayesian linear regression')
    
    # Plot the draw of data points
    for n in range(N):
        ax_number.scatter(x[n], t[n], facecolors='none', edgecolors='blue')

    # Plot the curve
    ax_number.plot(x_sin, y_sin, color='green')

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
    plotMCMCModel(x, t, N, x_sin, y_sin)

    plt.show(block=True)

if __name__ == "__main__":
    main()