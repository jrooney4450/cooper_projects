import numpy as np
import scipy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


if __name__ == "__main__":
    ########################### 1D ##################################
    
    # Generate draws from three gaussians
    mu1 = np.array([-1, -1]) 
    mu2 = np.array([1, 1])
    mu3 = np.array([1, -1])

    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[1, 0], [0, 1]])
    cov3 = np.array([[1, 0], [0, 1]])

    beta = 0.05
    
    x = np.linspace(-1,1,5)

    data1 = np.random.multivariate_normal(mu1, beta*cov1, size=x.shape)
    data2 = np.random.multivariate_normal(mu2, beta*cov2, size=x.shape)
    data3 = np.random.multivariate_normal(mu3, beta*cov3, size=x.shape)

    # print(mu1.reshape(-1,1))

    # plot some histograms
    plt.figure(1)
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Expectation Maximization 2D')

    plt.scatter(data1[:,0], data1[:,1], color = 'r')
    plt.scatter(data2[:,0], data2[:,1], color = 'b')
    plt.scatter(data3[:,0], data3[:,1], color = 'g')

    # # Plot as seperate datasets
    # no_bins = 150


    # # Plot as single data draw, infer seperate gaussians
    data = np.vstack((data1, data2, data3))

    # # count, bins, ignored = plt.hist(data, no_bins, density=True) 

    K = 3
    M = data.shape[1]
    N = data.shape[0]

    # print(M)

    # Mus are stored in K x M matrix
    # mus_old = np.zeros((K, M))
    mus_old = np.array([[-1, 1], [-1, 1], [-1, -1]])
    
    # covariances are stored in (M*K) x M matrix
    cov_old = np.eye((M))
    cov_old = np.vstack((cov_old, cov_old, cov_old))
    
    # Pis are stored in K-length array
    pis_old = np.array([0.3, 0.3, 0.4])

    # print(cov_old)

    # # Initialize with zero - will be overwritten immeditately
    mus_new = np.zeros((K, M))
    cov_new = np.zeros((M, M))
    cov_old = np.vstack((cov_old, cov_old, cov_old))
    pis_new = np.array([0, 0, 0])

    # print(data[0,:])
    # print(mu1)
    # print(cov1)
    # prob = multivariate_normal.pdf(np.array([-0.7, -0.5]), mu1, beta*cov1)
    # print(prob)
    # # print(pis_old[0])

    # # k = 0
    # j = 1
    # print(mus_old[j, :])
    # print(cov_old[j:j+M,:])
    # print(pis_old[j])
    # print(data[7,:])
    # sol = pis_old[j] * multivariate_normal.pdf([0, 0], mus_old[j, :], cov_old[j*2:j*2+M,:])
    # print(sol)

    for i in range(5): # TODO make a while loop to check proper convergence criterion
        # print('start of outer loop i = {}'.format(i))
        for k in range(K):
            # print('start of k loop k = {}'.format(k))
            mu_count = np.zeros((M))
            sig_count = np.eye((M))
            N_k_count = 0
            # print(mu_count)
            # print(sig_count)
            for n in range(N):
                # Eq. 9.23 - E step - Re-estimate responsibilities using new parameters
                x = data[n, :]
                resp_num = pis_old[k] * multivariate_normal.pdf(x, mus_old[k, :], cov_old[k*2:k*2+M,:])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * multivariate_normal.pdf(x, mus_old[j, :], cov_old[j*2:j*2+M,:])
                resp = resp_num / resp_den

                # M step - Re-estimate parameters using new responsibilites
                mu_count += resp * x
                N_k_count += resp # Eq. 9.27

            mus_new[k, :] = (1 / N_k_count) * mu_count # Eq. 9.24
            pis_new[k] = N_k_count / N # Eq. 9.26
            
            for n in range(N):
                # Eq. 9.23 - E step - Re-estimate responsibilities using new parameters
                x = data[n, :]
                resp_num = pis_old[k] * multivariate_normal.pdf(x, mus_old[k, :], cov_old[k*2:k*2+M,:])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * multivariate_normal.pdf(x, mus_old[j, :], cov_old[j*2:j*2+M,:])
                resp = resp_num / resp_den
                sig_count += resp * np.matmul((x.reshape(-1, 1) - mus_new[k, :].reshape(-1, 1)), (x.reshape(-1, 1) - (mus_new[k, :].reshape(-1, 1))).T)
            cov_new[k*2:k*2+M,:] = (1 / N_k_count) * sig_count # Eq. 9.28

        mus_old = mus_new; cov_old = cov_new; pis_old = pis_new

        print('k = {}, Mus {} \nSigs {} \nPis {}'.format(k, mus_old, cov_old, pis_old))
        # print("Mus {}, Sigs {}, Pis {}".format(mus, sigs, pis))

    # plt.show()