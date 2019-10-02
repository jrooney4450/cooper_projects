import math
import numpy as np
from scipy.stats import bernoulli,beta
from scipy.special import gamma
import scipy
import matplotlib.pyplot as plt
import collections

def plotML(p, x_array, sample_size, color1):

    x_sum = 0
    N = 1
    sq_err = 0

    for x in np.nditer(x_array):
        x_sum += x
        N += 1
        mu_ml = (1/N)*x_sum
        sq_err += (mu_ml-p)**2  
        mse_ml = sq_err/N
        plt.figure(1)
        plt.plot(N, mse_ml, '.', color=color1)
    
    return print("The probability of getting 1 with an ml approach is: {}".format(mu_ml))

def plotBayes(p, x_array, a, b, sample_size, color1):

    x_sum = 0
    N = 1
    sq_err = 0
    mu_bayes = a/(a+b) # initial mean of guess, or the prior

    for x in np.nditer(x_array):
        x_sum += x
        N += 1
        # Bayesian estimate
        m = x_sum # number of heads flips
        l = N - m # number of heads flips
        mu_bayes = (a+m)/(a+m+b+l) # update the mean based on the data
        sq_err += (mu_bayes-p)**2
        mse_bayes = sq_err/N
        plt.figure(1)
        plt.plot(N, mse_bayes, '.', color=color1)    

        # mu_bayes = (mu_bayes**(m+a-1))*((1-mu_bayes)**(l+b-1))
        # print(mu_bayes)
        # mu_bayes = (gamma(m+a+l+b)/(gamma(m+a)*gamma(l+b)))*(mu_ml**(m+a+l))*(1-mu_ml)**(l+b-1)
        # posterior_distro = mu_ml**()
        # plt.plot(N, mu_bayes, '.', color='black')

    return print("The probability of getting 1 with a Bayesian approach is: {} with hyperparameters a = {}, b = {}".format(mu_bayes,a,b))

    # likelihood_function1 = np.math.factorial(N)/np.math.factorial(N-m)!*np.math.factorial(m)
    # likelihood_function = likelihood_function*()

def plotProbDens(a, b):

    plt.figure(2)
    mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    plt.plot(x, beta.pdf(x, a, b),'r-', lw=1, alpha=0.6, label='beta pdf')
    return

if __name__ == "__main__":
    p = 0.5
    sample_size = 50
    x_brv_array = bernoulli.rvs(p, size=sample_size)
    
    # ML answer
    plotML(p, x_brv_array, sample_size, 'black')
    # good guess
    plotBayes(p, x_brv_array, 6, 5, sample_size, 'blue')
    # bad guess
    plotBayes(p, x_brv_array, 4, 40, sample_size, 'red')
    print("The actual probability is: {}\n\
The sample size is: {}\n".format(p,sample_size))
    plotProbDens(4, 40)

    plt.show()
    # m = 40
    # l = 35
    # mu_ml = 0.5
    # p_bayes = (gamma(m+a+l+b)/(gamma(m+a)*gamma(l+b)))*(mu_ml**(m+a+l))*((1-mu_ml)**(l+b-1))
    # print(gamma(m+a+l+b))
    # print(gamma(m+a)*gamma(l+b))
    # print(gamma(m+a+l+b)/gamma(m+a)*gamma(l+b))
    # print(p_bayes)

    # x_beta_array = beta.rvs(a, b, size=100)  # generates random observations in a beta distribution

