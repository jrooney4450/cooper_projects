import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


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

    # plot some histograms
    plt.figure(1)
    # plt.xlim(-0.01, 1)
    # plt.ylim(0.0, 1.01)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Expectation Maximization PDF over Histogram')

    # Plot as seperate datasets
    no_bins = 150
    # bin1 = int(sig1*no_bins); bin2 = int(sig2*no_bins); bin3 = int(sig3*no_bins)
    # count, bins, ignored = plt.hist(data1, bin1, density=True)
    # plt.plot(bins, 1/(sig1 * np.sqrt(2 * np.pi)) * \
    #     np.exp( - (bins - mu1)**2 / (2 * sig1**2) ), linewidth=2, color='r')
    # count, bins, ignored = plt.hist(data2, bin2, density=True)
    # plt.plot(bins, 1/(sig2 * np.sqrt(2 * np.pi)) * \
    #     np.exp( - (bins - mu2)**2 / (2 * sig2**2) ), linewidth=2, color='r')
    # count, bins, ignored = plt.hist(data3, bin3, density=True)
    # plt.plot(bins, 1/(sig3 * np.sqrt(2 * np.pi)) * \
    #     np.exp( - (bins - mu3)**2 / (2 * sig3**2) ), linewidth=2, color='r')

    # Plot as single data draw, infer seperate gaussians
    data = np.hstack((data1, data2, data3))

    # count, bins, ignored = plt.hist(data, no_bins, density=True) 



    K = 3
    N = data.shape[0]

    # Initialize guesses for values we want to obtain
    mus_old = [0, 0, 0]
    sigs_old = [0.1, 0.1, 0.1]
    pis_old = [0.33, 0.33, 0.34]

    mus_new = mus_old
    sigs_new = sigs_old
    pis_new = pis_old

    for i in range(1): # TODO make a while loop to check proper convergence criterion
        for k in range(K):
            mu_count = 0
            sig_count = 0
            N_k_count = 0
            N_k = 0
            for x in np.nditer(data):
                # Eq. 9.23 - E step - Re-estimate responsibilities using new parameters
                resp_num = pis_old[k] * mlab.normpdf(x, mus_old[k], sigs_old[k])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * mlab.normpdf(x, mus_old[j], sigs_old[j])
                resp = resp_num / resp_den

                # M step - Re-estimate parameters using new responsibilites
                mu_count += resp * x
                N_k_count += resp
            N_k = N_k_count / N # Eq. 9.27

            mus_new[k] = (1 / N_k) * mu_count # Eq. 9.24
            pis_new[k] = N_k / N # Eq. 9.26
            
            for x in np.nditer(data):
                sig_count += resp * (x - mus_new[k]) * (x - mus_new[k])
            sigs_new[k] = (1 / N_k) * sig_count # Eq. 9.28

            mus_old = mus_new; sigs_old = sigs_new; pis_old = pis_new
        
        print("Mus {}, Sigs {}, Pis {}".format(mus_old, sigs_old, pis_old))

            # Evaluate log likelihood

            

            


        # print("Mus {}, Sigs {}, Pis {}".format(mus, sigs, pis))

    

    for k in range(K):
        count, bins, ignored = plt.hist(data, no_bins, density=True)
        plt.plot(bins, 1/(sigs_new[k] * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mus_new[k])**2 / (2 * sigs_new[k]**2) ), linewidth=2, color='r')

    # plt.show()