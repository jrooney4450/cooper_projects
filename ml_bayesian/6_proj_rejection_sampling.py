import numpy as np
import scipy
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def main():
    # Generate three gaussians to form GMM
    K = 3
    mu1 = -1.9; mu2 = 0.2; mu3 = 2.1
    mus_true = [mu1, mu2, mu3]
    sig1 = 0.8; sig2 = 0.7; sig3 = 0.8
    sigs_true = [sig1, sig2, sig3]

    # Initialize plots
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Rejection Sampling to find a GMM Distribution')
    colors = ['tomato', 'slateblue', 'limegreen']
    labels = ['Gaussian 1', 'Gaussian 2', 'Gaussian 3']

    # Plot GMM distributions with histograms
    for k in range(K):  
        x = np.linspace(mus_true[k] - 3*sigs_true[k], mus_true[k] + 3*sigs_true[k], 100)
        plt.plot(x, norm.pdf(x, mus_true[k], sigs_true[k]), c=colors[k], label=labels[k])

    # Big boi gaussian that covers all of GMM: k * q(z)
    mu = 0.0; sig = 2.0; k = 6 
    x = np.linspace(mu - 3*sig, mu + 3*sig, 100)
    plt.plot(x, k * norm.pdf(x, mu, sig), label='kq(z)')

    # Generate data with the big boi gaussian
    N = 10000
    data = np.random.normal(mu, sig, N)
    data_new = []

    # Rejection sampling algorithm, section 11.1.2
    for pt in data:
        kq_pt = k * norm.pdf(pt, mu, sig) # max probabilty of particular data draw
        u = np.random.uniform(0, kq_pt) # pick u from uniform distribution
        if u > norm.pdf(pt, mu1, sig1) and u > norm.pdf(pt, mu2, sig2) and u > norm.pdf(pt, mu3, sig3):
            # Reject this point and plot success as red
            plt.scatter(pt, u, c='r', s=0.8)
        else:
            # Keep this point and plot success as green
            plt.scatter(pt, u, c='g', s=0.8)
            data_new.append(pt)

    # Plot the resulting data draw as a histogram
    bins = 30
    plt.hist(data_new, bins, density=True, color='mistyrose', label='new data draw')

    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()