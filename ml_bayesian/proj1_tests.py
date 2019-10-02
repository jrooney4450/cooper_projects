import math
import numpy as np
from scipy.stats import bernoulli,beta,gamma
import scipy
import matplotlib.pyplot as plt
import collections

def plotMLBernoulli(p, x_array, color1):

    # initialize loop variables
    x_sum = 0
    N = 1
    sq_err = 0

    # iterate through data and update the ML approximation
    for x in np.nditer(x_array):
        x_sum += x
        mu_ml = (1/N)*x_sum # equation 2.8 from Bishop textbook
        sq_err += np.square(mu_ml-p) # accumulate the error compared to the groud truth
        mse_ml = sq_err/N # formulation for obtaining the mean squared error
        plt.figure(1)
        plt.plot(N, mse_ml, '.', color=color1) # plot the MSE of each data point
        N += 1 # increment the data point counter


    return print("The probability of getting 1 with a ML approach is: {}".format(mu_ml))

def plotBayesBernoulli(p, x_array, a, b, color1, label):

    # initialize loop parameters
    x_sum = 0
    N = 1
    sq_err = 0

    mu_bayes = a/(a+b) # initial mean of guess through beta function hyperparameters or the prior

    # iterate through the data while applying the prior in order to make a Bayesian inference on the posterior
    for x in np.nditer(x_array):
        x_sum += x # accumulate the guesses into a single counter
        m = x_sum # number of heads flips
        l = N - m # number of tails flips
        mu_bayes = (a+m)/(a+m+b+l) # update the mean based on the data
        sq_err += (mu_bayes-p)**2 # accumulate the error compared to the groud truth
        mse_bayes = sq_err/N # calculate the MSE to this point in the dataset
        plt.figure(1) # plot the data onto the same figure as the ML guess
        plt.plot(N, mse_bayes, '.', color=color1) # plot the MSE of each data point
        N += 1 # increment the data point counter

    return print("The probability of getting 1 with a Bayesian approach is: {} with hyperparameters a = {}, b = {}".format(mu_bayes,a,b))

def plotBetaDistribution(a, b, x_array, title):

    # initialize loop parameters
    x_sum = 0
    N = 1

    for x in np.nditer(x_array):
        x_sum += x
        m = x_sum # number of heads flips
        l = N - m # number of tails flips
        N += 1 # increment the data point counter

    x = np.linspace(beta.ppf(0, a, b), beta.ppf(1, a, b), 1000) # generate x axis values for plotting the PDF
    
    # plot 2x2 subplots with the same x-axis and y-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    beta.stats(a, b, moments='mvsk')

    # first subplot is the beta distribution of the initial hyperparameters
    ax1.plot(x, beta.pdf(x, a, b))
    ax1.set_title(title + 'Initial Guess')

    # second subplot has 1/3 of the data accounted for
    ax2.plot(x, beta.pdf(x, a + int(m*(1/3)), b + int(l*(1/3))))
    ax2.set_title('Through 1/3 Data')
    
    # third subplot has 2/3 of the data accounted for
    ax3.plot(x, beta.pdf(x, a + int(m*(2/3)), b + int(l*(2/3))))
    ax3.set_title('Through 2/3 Data')
    
    # fourth subplot has all of the data
    ax4.plot(x, beta.pdf(x, a+m, b+l))
    ax4.set_title("Through all Data")
        
    return

def plotMLGauss(mu, sigma, x_array, color1):

    # initialize loop variables
    mu_sum = 0
    N = 1
    sq_err = 0

    for x in np.nditer(x_array):
        mu_sum += x # accumulate observations
        mu_ml = (1/N)*mu_sum # maximum likelihood estimator of the mean, Eq. 2.143
        sq_err += np.square(mu_ml-mu) # determine the error based on the ground truth mean
        mse_ml = sq_err/N # formulation of MSE
        plt.figure(4) # accrue into same figure as bayesian versions
        plt.plot(N, mse_ml, '.', color=color1, label='ML')
        N += 1 # increment the data point counter
    
    return print("The mean with a ML approach is: {}".format(mu_ml))


def plotBayesGauss(mu, sigma, a, b, x_array, color1, title):
    
    # initialize loop variables with prior
    mu_sum = a/b # Eq. 2.147
    var_sum = a/np.square(b) # Eq. 2.148
    N = 1
    sq_err = 0
    a_n = a
    b_n = b

    for x in np.nditer(x_array):
        var_sum += np.square(x - mu) # how far is the noisy observation from the known mean, squared
        var_bayes = (1/N)*var_sum # the maximum likelihood estimator of the variance
        b_n += var_bayes # Update the hyperparameter per Eq. 2.150
        mu_sum += x # offset of noisy observation from the known mean, then squared
        mu_bayes = (1/N)*mu_sum # maximum likelihood estimator of the mean
        a_n += N/2 # Update the hyperparameter per Eq. 2.151
        sq_err += np.square(mu_bayes-mu) # determine the error based on the ground truth mean
        mse_ml = sq_err/N # final MSE approximation
        plt.figure(4) # accrue into same figure as bayesian versions
        plt.plot(N, mse_ml, '.', color=color1, label=title+" Bayes")
        N += 1 # increment the data point counter
    
    return print("The variance with a Bayesian approach is: {} with hyperparameters a = {} and b = {}".format(mu_bayes,a,b))

def plotGammaDistributions(a, b, x_array, title):

    # initialize loop variables with prior
    mu_sum = a/b # Eq. 2.147
    var_sum = a/np.square(b) # Eq. 2.148
    N = 1
    sq_err = 0
    a_n = a
    b_n = b

    for x in np.nditer(x_array):
        var_sum += np.square(x - mu) # how far is the noisy observation from the known mean, squared
        var_bayes = (1/N)*var_sum # the maximum likelihood estimator of the variance
        b_n += var_bayes # Update the hyperparameter per Eq. 2.150
        mu_sum += x # offset of noisy observation from the known mean, then squared
        mu_bayes = (1/N)*mu_sum # maximum likelihood estimator of the mean
        a_n += N/2 # Update the hyperparameter per Eq. 2.151
        sq_err += np.square(mu_bayes-mu) # determine the error based on the ground truth mean
        mse_ml = sq_err/N # final MSE approximation
        plt.figure(4) # accrue into same figure as bayesian versions
        plt.plot(N, mse_ml, '.', label=title+" Bayes")
        N += 1 # increment the data point counter

    # plot 2x2 subplots with the same x-axis and y-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    gamma.stats(a, moments='mvsk')
    x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a_n), 1000) # generate x axis values for plotting the PDF

    # first subplot is the beta distribution of the initial hyperparameters
    ax1.plot(x, gamma.pdf(x, a, loc=(1/b)))
    ax1.set_title(title + 'Initial Guess')

    # second subplot has 1/3 of the data accounted for
    ax2.plot(x, gamma.pdf(x, a_n*(1/3)))
    ax2.set_title('Through 1/3 Data')
    
    # third subplot has 2/3 of the data accounted for
    ax3.plot(x, gamma.pdf(x, a_n*(2/3)))
    ax3.set_title('Through 2/3 Data')
    
    # fourth subplot has all of the data
    ax4.plot(x, gamma.pdf(x, a_n, loc=(1/b_n)))
    ax4.set_title("Through all Data")

    return

if __name__ == "__main__":
    
    ################################# Start of first simulation (Bernouilli)

    sample_size = 100 # Number of observations in data set

#     p = 0.5 # Probability of obtaining a 1 response
#     x_brv_array = bernoulli.rvs(p, size=sample_size) # generate bernouilli random values
    
#     # Format the first plot of MSE vs. N
#     plt.figure(1)
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.xlabel('Number of Data Points (N)')
#     plt.title('MSE vs. N')

#     # ML answer
#     plotMLBernoulli(p, x_brv_array, "black")
    
#     # Bayes good guess
#     plotBayesBernoulli(p, x_brv_array, 5, 5, "blue", "Good ")
    
#     # Bayes bad guess
#     plotBayesBernoulli(p, x_brv_array, 4, 40, "red", "Bad ")
    
#     # Bayes distribution with good guess
#     plotBetaDistribution(5, 5, x_brv_array, "Good ")

#     # Bayes distribution with bad guess
#     plotBetaDistribution(4, 40, x_brv_array, "Bad ")

#     print("The actual probability is: {}\n\
# The sample size is: {}\n".format(p,sample_size))

    ################################# Start of second simulation (Gaussian)

    mu, sigma = 70, 10 # parameters neede to define a normal distribution
    var = np.square(sigma) # determine the ground truth variance

    x_norm_array = np.random.normal(mu, sigma, sample_size) # generate random values within the normal distribution

    # Format the Gaussian plot of MSE vs. N
    plt.figure(4)
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Number of Data Points (N)')
    plt.title('Gaussian MSE vs. N')

    # plot MSE versus N of Gaussian with unknown variance, known mean
    plotMLGauss(mu, sigma, x_norm_array, "black")

    # good guess
    plotBayesGauss(mu, sigma, 10, 1, x_norm_array, "blue", "Good ")

    # bad guess
    plotBayesGauss(mu, sigma, 40, 1, x_norm_array, "red", "Bad ")

    # covergence of the gamma function based on a bad initial guess
    plotGammaDistributions(1, 1, x_norm_array, "Bad ")

    # For the plot below, gamma.pdf was used through the scipy kit. 
    # This particular implementation did not match Bishops methodology.
    # Namely, the b parameter is not taken into account. The mean should cener around 
    # 70 in this case, but it does not. One can still see the gradual movement
    # of datapoints to the correct postision, however, just the scale and 
    # implementation are incorrect

    print("The actual mean is: {}\n\
The sample size is: {}\n".format(mu,sample_size))

    plt.show()