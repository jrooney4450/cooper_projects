import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

if __name__ == "__main__":
    ########################### 1D ##################################
    
    # Generate draws from three gaussians
    mu1 = -0.8; mu2 = -0.2; mu3 = 0.4
    mus_true = [mu1, mu2, mu3]
    sig1 = 0.1; sig2 = 0.2; sig3 = 0.25
    sigs_true = [sig1, sig2, sig3]

    # Generate random draws of data following generated gaussians
    x = np.linspace(-1,1,200)
    data1 = np.random.normal(mu1, sig1, x.shape)
    data2 = np.random.normal(mu2, sig2, x.shape)
    data3 = np.random.normal(mu3, sig3, x.shape)

    # Combine to single data draw
    data = np.hstack((data1, data2, data3))

    # Initialize guesses for values we want to obtain
    mus_old = [-1, 0, 1]
    sigs_old = [1, 1, 1]
    pis_old = [0.33, 0.33, 0.34]

    print('Initial guess:\nMeans: \n{}\n Sigmas: \n{}\n'\
        .format(mus_old, sigs_old))

    # Initialize update parameters with zero - will be overwritten immeditately
    mus_new = [0, 0, 0]
    sigs_new = [0, 0, 0]
    pis_new = [0, 0, 0]

    # Initialize plots
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Expectation Maximization PDF over Histogram')
    no_bins = 100
    bin1 = int(sig1*no_bins); bin2 = int(sig2*no_bins); bin3 = int(sig3*no_bins)
    bin_list = [bin1, bin2, bin3]
    data_list = [data1, data2, data3]
    colors = ['r', 'b', 'g']
    colors2 = ['tomato', 'slateblue', 'limegreen']

    # Initialize loop variables
    K = 3
    N = data.shape[0]
    thresh = 0.01
    counter = 0
    ll_prev = 0
    ll_diff = 10
    while ll_diff > thresh: # Checks that the difference in the log-likelihood approaches zero
        
        # Make a movie! Plot histogram with pdf after each parameter update
        plt.figure(1)
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Expectation Maximization PDF over Histogram')

        k = 0
        count, bins, ignored = plt.hist(data_list[k], bin_list[k], density=True, color = colors[k])
        plt.plot(bins, 1/(sigs_old[k] * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mus_old[k])**2 / (2 * sigs_old[k]**2) ), linewidth=2, color = colors2[k])
        
        k = 1
        count, bins, ignored = plt.hist(data_list[k], bin_list[k], density=True, color = colors[k])
        plt.plot(bins, 1/(sigs_old[k] * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mus_old[k])**2 / (2 * sigs_old[k]**2) ), linewidth=2, color = colors2[k])
        
        k = 2
        count, bins, ignored = plt.hist(data_list[k], bin_list[k], density=True, color = colors[k])
        plt.plot(bins, 1/(sigs_old[k] * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mus_old[k])**2 / (2 * sigs_old[k]**2) ), linewidth=2, color = colors2[k])

        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
        
        for k in range(K):
            mu_count = 0
            N_k_count = 0
            for x in np.nditer(data):
                # Eq. 9.23 - E step - Find responsibility for particular data points
                resp_num = pis_old[k] * mlab.normpdf(x, mus_old[k], sigs_old[k])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * mlab.normpdf(x, mus_old[j], sigs_old[j])
                resp = resp_num / resp_den

                # M step - update parameters using responsibility
                mu_count += resp * x
                N_k_count += resp # Eq. 9.27

            # Calculate updated means and pis
            mus_new[k] = (1 / N_k_count) * mu_count # Eq. 9.24
            pis_new[k] = N_k_count / N # Eq. 9.26
            
            # New mean is needed for variance calculations
            # Re-calculate responsibility for Eq. 9.28 by looping through data again
            sig_count = 0
            for x in np.nditer(data):
                resp_num = pis_old[k] * mlab.normpdf(x, mus_old[k], sigs_old[k])
                resp_den = 0
                for j in range(K):
                    resp_den += pis_old[j] * mlab.normpdf(x, mus_old[j], sigs_old[j])
                resp = resp_num / resp_den
                sig_count += resp * (x - mus_new[k])**2

            sigs_new[k] = (1 / N_k_count) * sig_count # Eq. 9.28

            # Prevent covariance from approaching a singularity by smoothing the outliers
            if sigs_new[k] < 0.1:
                sigs_new[k] += 0.15
            elif sigs_new[k] > 1.0:
                sigs_new[k] -= 0.1

        # Evaluate the log likelihood - Eq. 9.28
        # Use to determine when EM cannot find a better solution
        ll = 0
        for x in np.nditer(data):
            ll_count = 0
            for k in range(K):
                ll_count += pis_old[k] * mlab.normpdf(x, mus_old[k], sigs_old[k])
            ll += np.log(ll_count)
        ll_diff = np.abs(ll - ll_prev)
        ll_prev = ll
        # print('log likelihood {}\nll difference {}'.format(ll, ll_diff))
        
        counter += 1
        
        # Reassign old parameters to new values and iterate algorithm
        mus_old = mus_new; sigs_old = sigs_new; pis_old = pis_new

    # Print restults to console
    print('Final values!\nActual Means: \n{}\nEM Means: \n{}\nActual Sigmas: \n{}\nEM Sigmas: \n{}\
        \nafter {} iterations'.format(mus_true, mus_new, sigs_true, sigs_new, counter))

    # Leave final plot open upon completion
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Expectation Maximization PDF over Histogram')
    for k in range(K):  
        count, bins, ignored = plt.hist(data_list[k], bin_list[k], density=True, color = colors[k])
        plt.plot(bins, 1/(sigs_new[k] * np.sqrt(2 * np.pi)) * \
            np.exp( - (bins - mus_new[k])**2 / (2 * sigs_new[k]**2) ), linewidth=2, color = colors2[k])

    plt.show(block=True)