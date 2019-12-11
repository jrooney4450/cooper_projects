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

    # Choose parameters
    M = 10 # Number of weights
    sig = 0.2
    beta = (1/sig)**2
    alpha = 2.0

    # Perform the Bayesian Linear Regression Treatment as was done for homework 2
    iota = np.zeros((N, M))
    for i in range(N):
        iota[i, :] = getPhi(M, x, x, i)
    t = t.reshape(-1, 1)
    S_N = np.linalg.inv(alpha*np.identity(M) + beta*(np.matmul(np.transpose(iota), iota))) # Eq. 3.54
    m_N = beta * np.matmul(np.matmul(S_N, np.transpose(iota)), t) # Eq. 3.53

    # Chose variance for prior for MCMC
    k = 1
    S0 = k * np.identity(M)

    # Choose variance for z_star updates for MCMC
    var = 0.2
    S = var * np.identity(M)

    # Generate initial guesses for z_prev and z_star
    z_prev = (np.random.random_sample(M) * 2) - 1
    z_star = np.random.multivariate_normal(z_prev, S)
    z = [z_prev, z_star] # store into list
    p = [0, 0] # store posterior probabilities of z_prev and z_star into list

    # Initialize plot
    fig, ax_loop = plt.subplots()
    ax_loop.set_title('MCMC Weight Pairs Accepted and Rejected')
    ax_loop.set_xlabel('w0')
    ax_loop.set_ylabel('w1') 

    # Initialize loop variables
    z_avg_sum = np.zeros((M,))
    avg_count = 0.0
    N_burn_in = 200
    N_loop = 2000
    isNotPrinted = True

    for count in range(N_loop):
        for k in range(2):
            like = 0
            # Eq. 3.10 - Calculate the likelihood
            for i in range(N):
                phi = getPhi(M, x, x, i)
                like += mlab.normpdf(t[i], np.dot(z[k], phi), var) # Eq. 3.10 for likelihood
            
            # Recenter proposal distribution (prior) around last accepted z_value
            prior = multivariate_normal.pdf(z[k], z[0], S0)
            
            # Calculate the log posterior
            p[k] = np.log(prior + like)

        # Confirm that the model has run through the burn in phase
        if count > N_burn_in and isNotPrinted:
            print('burn in phase completed!')
            isNotPrinted = False

        # Eq. 11.33 - Use metropolis criterion to accept or reject weight pairs
        u = np.random.uniform(0, 1)
        prob = p[1] / p[0]
        A = min([1.0, prob])
        if A > u: # Accept this point
            ax_loop.scatter(z[1][0], z[1][1], color = 'green', s=0.8) # plot z_star as success
            z[0] = z[1] # z_star assigned to z_prev
            z[1] = np.random.multivariate_normal(z[0], S) # draw another z_star value
            if count > N_burn_in: # Start averaging values after burn in complete
                z_avg_sum += z[0] # Sum the accepted z_star value
                avg_count += 1

        else: # Reject this point
            ax_loop.scatter(z[1][0], z[1][1], color = 'red', s=0.8) # plot z_star as failure
            z[1] = np.random.multivariate_normal(z[0], S) # try again with another data point

    best_z = z_avg_sum / (avg_count)

    print('bayesian old weights: {}'.format(m_N[:,0]))
    print('avg weight from mcmc: {}'.format(best_z))

    # Calculate the regression line for the MCMC weights
    I = 100
    x_plot = np.linspace(-0.1, 1.1, I)
    mean_list = []
    low_list = []
    high_list = []
    for i in range(I):
        design_vector = getPhi(M, x_plot, x, i).reshape(-1, 1)
        mean_arr = np.matmul(np.transpose(best_z), design_vector)
        mean_list.append(mean_arr[0])

    fig, ax2 = plt.subplots()
    ax2.plot(x_plot, mean_list, color='red')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('MCMC Linear Regression')

    # Plot the draw of data points
    for n in range(N):
        ax2.scatter(x[n], t[n], facecolors='none', edgecolors='blue')

    # Plot the original sine curve
    ax2.plot(x_sin, y_sin, color='green')

    # Calculate the regression line for bayesian estimate which is correct
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

    N = 25
    plotMCMCModel(x, t, N, x_sin, y_sin)

    plt.show(block=True)

if __name__ == "__main__":
    main()