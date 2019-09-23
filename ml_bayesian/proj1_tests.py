import math
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

p = 0.5 # probability of choosing 1
x_array = bernoulli.rvs(p, size=1000) # generate random bernouilli vars

# initialize needed loop variables
ml_sum = 0
ml_N = 1

# for each data point, recalculate the mu_ml and use to find the RMS error
for x in np.nditer(x_array):
    ml_sum += x
    ml_N += 1
    ml_mu = (1/ml_N)*ml_sum
    ml_rms = np.sqrt(np.square(ml_mu-p))
    plt.plot(ml_N, ml_rms, 'o', color='black')

plt.show()