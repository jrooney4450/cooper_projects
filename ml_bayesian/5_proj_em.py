import numpy as np
import scipy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ########################### 1D ##################################
    
    # Generate draws from three gaussians
    mu1 = -0.6; mu2 = 0.2; mu3 = 0.8
    sig1 = 0.15; sig2 = 0.4; sig3 = 0.1
    
    N = 50
    x = np.linspace(-1,1,300)

    data1 = np.random.normal(mu1, sig1, x.shape)
    data2 = np.random.normal(mu2, sig2, x.shape)
    data3 = np.random.normal(mu3, sig3, x.shape)

    plt.figure(1)
    # plt.xlim(-0.01, 1)
    # plt.ylim(0.0, 1.01)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Expectation Maximization PDF over Histogram')

    # plt.scatter(x1, data1, color='r')
    # plt.scatter(x2, data2, color='b')
    # plt.scatter(x3, data3, color='g')

    mult = 150
    bin1 = int(sig1*mult); bin2 = int(sig2*mult); bin3 = int(sig3*mult)

    count, bins, ignored = plt.hist(data1, bin1, density=True)
    plt.plot(bins, 1/(sig1 * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu1)**2 / (2 * sig1**2) ), linewidth=2, color='r')

    count, bins, ignored = plt.hist(data2, bin2, density=True)
    plt.plot(bins, 1/(sig2 * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu2)**2 / (2 * sig2**2) ), linewidth=2, color='r')

    count, bins, ignored = plt.hist(data3, bin3, density=True)
    plt.plot(bins, 1/(sig3 * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu3)**2 / (2 * sig3**2) ), linewidth=2, color='r')


    plt.show()