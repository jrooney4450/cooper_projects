import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp

def basFunGauss(input_x, mu):
    # Using a gaussian basis function with mu's of random x value draws
    phi_of_x = (1 / noise_sigma*2*np.pi**(1/2)) * np.exp((-((input_x-mu)**2)/(2*noise_sigma**2)))
    return phi_of_x

def gaussKernel(x, x_prime):
    sig = 1
    k_of_x = (1 / sig*2*np.pi**(1/2)) * np.exp(np.linalg.norm(x - x_prime) / (2 * sig**2))
    return k_of_x

def linearKernel(x, x_prime):
    # return x * x_prime
    sig = 1
    k_of_x = (1 / sig*2*np.pi**(1/2)) * np.exp(np.linalg.norm(x - x_prime) / (2 * sig**2))


    return k_of_x

def plotModel(ax_number, N, title):
    # Plot original sine curve
    ax_number.plot(x_sin, y_sin, color='green')

    # Create the target vector based on the data draw and use find the new mean of the weights
    target_vector = np.array([[target[0]]])
    for i in range(N-1):

        add_me = np.array([[target[i+1]]])
        target_vector = np.concatenate((target_vector, add_me), axis=0)
    # print(target_vector)

    # Construct the covariance matrix per Eq. 6.62
    C = np.zeros((N,N))
    for n in range(N):
        for m in range(N):
            C[n,m] = linearKernel(x[n], x[m])
            # print('xn is {}, xm is {} which reults in {}'.format(x[n], x[m], C[n,m]))
    delta = np.eye(N)
    C = C + ((1/beta) * delta)
    # print(C)

    # Find means 
    N_plot = 100
    x_list = np.linspace(0, 1, N_plot)
    # x_list = np.random.uniform(0, 1, N_plot)
    # print(x_list)
    c = np.zeros((1,1))

    k = np.zeros((N, 1))
    for j in range(N):
        k[j, :] = linearKernel(x[j], x_list[0])
    # k = np.vstack((k, linearKernel(x[N], x_list[0])))
    # print(k)

    # Find lots of means and plot
    mean_list = []
    mean_low = []
    mean_high = []
    # print(C.shape)
    for i in range(len(x_list) - 1):
        # print('This is the {} loop'.format(i))
        # print(k)
        # print(C)
        # print(target_vector)
        # m_next = np.matmul(np.matmul(k.T, np.linalg.inv(C)), target_vector) # Eq. 6.66
        C_inv = np.linalg.inv(C)
        m_next = np.matmul(k.T, C_inv)
        m_next = np.matmul(m_next, k)
        print(m_next)
        mean_list.append(m_next[0,0])

        c[0,0] = linearKernel(x_list[i], x_list[i]) + (1/beta)
        covar_next = np.matmul(k.T, C_inv) # Eq. 6.67
        covar_next = c - np.matmul(covar_next, k)
        # print(covar_next[0,0])
        
        mean_low.append(m_next[0,0] - np.sqrt(covar_next[0,0]))
        mean_high.append(m_next[0,0] + np.sqrt(covar_next[0,0]))
        
        dummy1 = np.hstack((C, k))
        # print(C.shape)
        # print(k.shape)
        # print(dummy1.shape)
        dummy2 = np.hstack((k.T, c))
        # print(dummy2.shape)
        C = np.vstack((dummy1, dummy2))
        # print(C.shape)
        target_vector = np.vstack((target_vector, m_next))
        # print(target_vector)
        k = np.zeros((N+i+1, 1))
        for j in range(N):
            k[j, :] = linearKernel(x[j], x_list[i+1])
            # print(k)
        for l in range(i+1):
            k[N+l, :] = linearKernel(x_list[l], x_list[i+1])
            # print(k)
        # print(k)
    
    # print(x_list[:-1].shape)
    # print(len(mean_list))
    ax_number.plot(x_list[:-1], mean_list, color = 'r')
    ax_number.fill_between(x_list[:-1], mean_low, mean_high, color='mistyrose')
    # ax_number.set_xlabel('x')
    # ax_number.set_ylabel('t')
    # ax_number.set_title(title)

    # Plot the draw of data points
    for n in range(N):
        ax_number.scatter(x[n], target[n], facecolors='none', edgecolors='blue')
    return

if __name__ == "__main__":

    # Generate data points for the original sine curve
    x_sin = np.linspace(0, 1, 500)
    y_sin = np.sin(math.pi*2*x_sin)

    # Get a random draw of data
    N = 26
    x = np.random.random_sample(200,)
    # Ensure two of the data points occur at the limits
    # x[3] = 0
    # x[4] = 1
    noise_sigma = 0.2
    noise_mean = 0
    noise_t = np.random.normal(noise_mean, noise_sigma, x.shape)
    target = np.sin(math.pi*2*x) # find true value of target
    target += noise_t # Eq. 6.57 add gaussian noise to target

    # N_train = 20
    # X_tr = np.array([x[:N_train]]).reshape(-1,1)
    # y_tr = np.array([target[:N_train]]).reshape(-1,1)
    # X_te = np.array([x[N_train:N_train+5]]).reshape(-1,1)
    # y_tr = np.array([target[N_train:N_train+5]]).reshape(-1,1)

    # kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    # model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    # model.fit(X_tr, y_tr)
    # params = model.kernel_.get_params()
    # y_pred, std = model.predict(X_te, return_std=True)


    # # Construct the prior
    beta = (1/noise_sigma)**2
    alpha = 2.0
    # m_0 = [0, 0]
    # S_0 = alpha**(-1)*np.identity(2)

    # plot 2x2 subplots with the same x-axis and y-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    # ax1.set_xlim(left = -0.1, right = 1.1)
    # ax1.set_ylim(bottom = -1.2, top = 1.2)

    # Run the model with seperate draws of data
    # plotModel(ax1, 1, "one data point")
    # plotModel(ax2, 2, "two data points")
    # plotModel(ax3, 4, "four data points")
    plotModel(ax4, 7, "7 data points")

    plt.show()