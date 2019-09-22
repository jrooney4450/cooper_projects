import math
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

p = 0.3 # probability of choosing 1
flips = bernoulli.rvs(p, size=1000) # generate random bernouilli vars

sum_x = 0

N = flips.size

for flip in np.nditer(flips):
    sum_x += flip

mu_ml = (1/N)*sum_x
print(mu_ml)