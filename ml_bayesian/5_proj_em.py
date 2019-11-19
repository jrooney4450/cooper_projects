import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


if __name__ == "__main__":
    ########################### 1D ##################################
    
    # Generate draws from three gaussians
    mu1 = -2.; mu2 = -0.1; mu3 = 1.5
    mus_true = [mu1, mu2, mu3]
    sig1 = 0.5; sig2 = 0.4; sig3 = 0.7
    sigs_true = [sig1, sig2, sig3]

    x = np.linspace(-1,1,150)

    data1 = np.random.normal(mu1, sig1, x.shape)
    data2 = np.random.normal(mu2, sig2, x.shape)
    data3 = np.random.normal(mu3, sig3, x.shape)

    # Combine to single data draw, infer seperate gaussians
    data = np.hstack((data1, data2, data3))

    # count, bins, ignored = plt.hist(data, no_bins, density=True) 

    K = 3
    N = data.shape[0]

    # Initialize guesses for values we want to obtain
    mus_old = [-1, 0, 1]
    sigs_old = [1, 1, 1]
    pis_old = [0.33, 0.33, 0.34]

    # Initialize update parameters with zero - will be overwritten immeditately
    mus_new = [0, 0, 0]
    sigs_new = [0, 0, 0]
    pis_new = [0, 0, 0]

    diff = 10.
    thresh = 0.00005
    counter = 0
    while diff > thresh: # TODO make a while loop to check proper convergence criterion
        # print('start of outer loop i = {}'.format(i))
        for k in range(K):
            # print('start of k loop k = {}'.format(k))
            mu_count = 0
            sig_count = 0
            N_k_count = 0
            for x in np.nditer(data):
                # Eq. 9.23 - E step - Re-estimate responsibilities using new parameters
                resp_num = pis_old[k] * mlab.normpdf(x, mus_old[k], sigs_old[k])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * mlab.normpdf(x, mus_old[j], sigs_old[j])
                    # print('resp_den = {}'.format(resp_den))
                resp = resp_num / resp_den
                # print('x = {}, k = {}, resp = {}'.format(x, k, resp))

                # M step - Re-estimate parameters using new responsibilites
                mu_count += resp * x
                N_k_count += resp # Eq. 9.27

            mus_new[k] = (1 / N_k_count) * mu_count # Eq. 9.24
            pis_new[k] = N_k_count / N # Eq. 9.26
            
            for x in np.nditer(data):
                resp_num = pis_old[k] * mlab.normpdf(x, mus_old[k], sigs_old[k])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * mlab.normpdf(x, mus_old[j], sigs_old[j])
                resp = resp_num / resp_den
                sig_count += resp * (x - mus_new[k]) * (x - mus_new[k])
            sigs_new[k] = (1 / N_k_count) * sig_count # Eq. 9.28

            # My sigmas kept diverging to zero - this prevents that from happening
            if sigs_new[k] < 0.1:
                sigs_new[k] = 0.1
            diff = np.abs(mus_new[k] - mus_old[k])
            print(diff)
            counter += 1

        mus_old = mus_new; sigs_old = sigs_new; pis_old = pis_new

        # print('k = {}, Mus {}, Sigs {}, Pis {}'.format(k, mus_old, sigs_old, pis_old))

    print('Final values!\nActual Means: \n{}\nEM Means: \n{}\nActual Sigmas: \n{}\nEM Sigmas: \n{}\
        \nafter {} iterations'.format(mus_true, mus_new, sigs_true, sigs_new, counter))

    # plot some histograms
    plt.figure(1)
    # plt.xlim(-0.01, 1)
    # plt.ylim(0.0, 1.01)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Expectation Maximization PDF over Histogram')

    # Plot as seperate datasets
    no_bins = 50
    bin1 = int(sig1*no_bins); bin2 = int(sig2*no_bins); bin3 = int(sig3*no_bins)
    # count, bins, ignored = plt.hist(data1, bin1, density=True)
    # plt.plot(bins, 1/(sig1 * np.sqrt(2 * np.pi)) * \
    #     np.exp( - (bins - mu1)**2 / (2 * sig1**2) ), linewidth=2, color='r')
    # count, bins, ignored = plt.hist(data2, bin2, density=True)
    # plt.plot(bins, 1/(sig2 * np.sqrt(2 * np.pi)) * \
    #     np.exp( - (bins - mu2)**2 / (2 * sig2**2) ), linewidth=2, color='r')
    # count, bins, ignored = plt.hist(data3, bin3, density=True)
    # plt.plot(bins, 1/(sig3 * np.sqrt(2 * np.pi)) * \
    #     np.exp( - (bins - mu3)**2 / (2 * sig3**2) ), linewidth=2, color='r')

    # Plot as seperate datasets
    no_bins = 50
    bin1 = int(sig1*no_bins); bin2 = int(sig2*no_bins); bin3 = int(sig3*no_bins)

    count, bins, ignored = plt.hist(data1, bin1, density=True, color = 'r')
    plt.plot(bins, 1/(sig1 * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu1)**2 / (2 * sig1**2) ), linewidth=2, color='r')

    count, bins, ignored = plt.hist(data2, bin2, density=True, color = 'g')
    plt.plot(bins, 1/(sig2 * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu2)**2 / (2 * sig2**2) ), linewidth=2, color='r')

    count, bins, ignored = plt.hist(data3, bin3, density=True, color = 'b')
    plt.plot(bins, 1/(sig3 * np.sqrt(2 * np.pi)) * \
        np.exp( - (bins - mu3)**2 / (2 * sig3**2) ), linewidth=2, color='r')

    plt.show()