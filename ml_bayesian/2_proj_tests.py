import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

a0 = -0.3
a1 = 0.5
N = 20 # sample size
x = 2 * np.random.random_sample(N,) - 1
# print(x)

noise_sigma = 0.2
noise_mean = 0
noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
target = a0 + a1*x # find true value of target
target += noise_t # Eq. 3.7 add gaussian noise to target
# print(target)
# print(noise_t)
# print(target + noise_t)

# Parameters on prior
w0_mu = 0
w0_sigma = 0.5
w0_variance = np.square(w0_sigma)

w1_mu = 0
w1_sigma = 0.5
w1_variance = np.square(w1_sigma)

# Create grid and multivariate normal
x_graphing = np.linspace(-1,1,500)
y_graphing = np.linspace(-1,1,500)
X, Y = np.meshgrid(x_graphing,y_graphing)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([w0_mu, w1_mu], [[w0_variance, 0], [0, w1_variance]])

# # Make a 3D plot of prior
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
# ax.set_xlabel('w0')
# ax.set_ylabel('w1')
# ax.set_zlabel('Z axis')

# Draw lines using w0 and w1 from prior
p_N = 6 # number of samples
w0 = np.random.normal(w0_mu, w0_sigma, p_N)
w1 = np.random.normal(w1_mu, w1_sigma, p_N)

# fig = plt.figure()
# ax = plt.axes()

# for n in range(w0.size):
#     ax.plot(x, w0[n] + w1[n]*x)

# plt.title("data space")
# plt.xlabel("x")
# plt.ylabel("y")

#### Attempt with first data point

# Use gaussian basis functions Eq. 3.4
# For much of the book the vector phi(x) = x


beta = 1/noise_sigma
alpha = 2.0

# Covariance matrix from prior
S_0 = (alpha**(-1)) * np.identity(2) 
iota = np.array([x[0], x[1]])
print(iota)

S_N_inverse = (alpha**(-1))*np.identity(2) + beta*(np.matmul(np.transpose(iota),iota)) # Eq. 3.54
# print(S_N_inverse)

m_N = np.matmul(beta*S_N_inverse,np.transpose(iota)*target[0])
print(m_N)

S_N = np.linalg.inv(S_N_inverse) # updated variance
print(S_N)

# get new posterior distribution
rv = multivariate_normal(m_N, S_N)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('Z axis')

plt.show()