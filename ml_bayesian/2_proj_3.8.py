import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt    

if __name__ == "__main__":

    # Plot the original sine curve
    fig = plt.figure()
    ax = plt.axes()
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(math.pi*2*x_sin)
    ax.plot(x_sin, y_sin, color='green')

    # Need data sets of N = 1, 2, 4, 25
    N = 25
    x = np.random.random_sample(N,)
    noise_sigma = 0.1
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = np.sin(math.pi*2*x) # find true value of target
    target += noise_t # Eq. 3.7 add gaussian noise to target
    print(target)
    ax.scatter(x, target, s=50, facecolors='none', edgecolor='blue', )


    # subplotting code
    # # plot 2x2 subplots with the same x-axis and y-axis
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    # beta.stats(a, b, moments='mvsk')

    # # first subplot is the beta distribution of the initial hyperparameters
    # ax1.plot(x, beta.pdf(x, a, b))
    # ax1.set_title(title + 'Initial Guess')

    # # second subplot has 1/3 of the data accounted for
    # ax2.plot(x, beta.pdf(x, a + int(m*(1/3)), b + int(l*(1/3))))
    # ax2.set_title('Through 1/3 Data')
    
    # # third subplot has 2/3 of the data accounted for
    # ax3.plot(x, beta.pdf(x, a + int(m*(2/3)), b + int(l*(2/3))))
    # ax3.set_title('Through 2/3 Data')
    
    # # fourth subplot has all of the data
    # ax4.plot(x, beta.pdf(x, a+m, b+l))
    # ax4.set_title("Through all Data")

    plt.show()