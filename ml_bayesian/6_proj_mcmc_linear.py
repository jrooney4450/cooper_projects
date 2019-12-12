import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def getPhi(M, x, i):
    phi = np.zeros(M,)
    phi[0] = 1
    for m in range(M - 1):
        phi[m + 1] = x[i]
    return phi

def plotMCMCModel(a, x, t, N, alpha, beta):
    x = x[0:N]
    t = t[0:N]
 
    # Generate initial guesses for z_prev and z_star
    M = 2
    S0 = 1/alpha*np.eye(2)
    S = S0/100
    z_prev = (np.random.random_sample(M) * 2) - 1
    z = [z_prev, z_prev] # store into list
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

    phi = np.zeros((N,2))
    phi[:,0] = 1
    phi[:,1] = x

    # Kick off simulation
    log_prior = np.log(multivariate_normal.pdf(z[0], z[0], S0))
    log_like = N * np.log(1 / np.sqrt(2 * np.pi / beta))
    for i in range(N):
        log_like += -1/2 * beta * (t[i] - np.matmul(z[0].T, phi[i,:]))**2

    p[0] = log_prior + log_like

    for count in range(N_loop):
        z[1] = np.random.multivariate_normal(z[0], S) # draw a z_star value

        # Eq. 3.10 - Calculate the likelihood of new guess
        log_like = N * np.log(1 / np.sqrt(2 * np.pi / beta))
        for i in range(N):
            log_like += -1/2 * beta * (t[i] - np.matmul(z[1].T, phi[i,:]))**2

        # Recenter proposal distribution (prior) around last accepted z_value
        log_prior = np.log(multivariate_normal.pdf(z[1], z[0], S0))
        
        # Calculate the log posterior
        p[1] = log_prior + log_like

        # Confirm that the model has run through the burn in phase
        if count > N_burn_in and isNotPrinted:
            print('burn in phase completed!')
            isNotPrinted = False

        # Eq. 11.33 - Use metropolis criterion to accept or reject weight pairs
        prob = p[1] - p[0]
        if (p[1] - p[0]) >= 0: # Accept this point
            ax_loop.scatter(z[1][0], z[1][1], color = 'green', s=0.8) # plot z_star as success
            z[0] = z[1] # z_star assigned to z_prev
            p[0] = p[1] # set new probability to previous
            if count > N_burn_in: # Start averaging values after burn in complete
                z_avg_sum += z[0] # Sum the accepted z_star value
                avg_count += 1.0
        else:
            u = np.random.uniform(0, 1)
            exp_prob = np.exp(prob)
            if u <= exp_prob: # Accept this point
                ax_loop.scatter(z[1][0], z[1][1], color = 'green', s=0.8) # plot z_star as success
                z[0] = z[1] # z_star assigned to z_prev
                p[0] = p[1] # set new probability to previous
                if count > N_burn_in: # Start averaging values after burn in complete
                    z_avg_sum += z[0] # Sum the accepted z_star value
                    avg_count += 1.0
            else: # Reject this point
                ax_loop.scatter(z[1][0], z[1][1], color = 'red', s=0.8) # plot z_star as failure

    best_z = z_avg_sum / (avg_count) # take average of all accepted mean pairs

    # Plot location of actual weights
    ax_loop.scatter(a[0], a[1], color='white', marker='x')

    print('synthetic weights: {}'.format(a))
    print('avg weights from mcmc: {}'.format(best_z))

def main():
    
    # True weights we wish to recover
    a = [0.5, -0.5]

    # Generate true data points from true weights
    N = 25
    x = 2 * np.random.random_sample(N,) - 1

    # Add gaussian noise to the true data points
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    t = a[0] + a[1]*x # find true value of target
    t += noise_t # Eq. 3.7 add gaussian noise to target

    # Generate a prior with a mean of 0 and diagonal covariance
    beta = (1/noise_sigma)**2
    alpha = 2.0

    plotMCMCModel(a, x, t, N, alpha, beta)

    plt.show()

if __name__ == "__main__":
    main()