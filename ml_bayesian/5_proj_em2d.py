import numpy as np
import scipy
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def plot_ellipse(ax, mu, sigma, color="k"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ax.tick_params(axis='both', which='major', labelsize=10)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse) 

if __name__ == "__main__":
    ########################### 2D ##################################
    
    # Generate true means and covariances of three 2D gaussians 
    mu1 = np.array([-0.66, 0]) 
    mu2 = np.array([1, 1])
    mu3 = np.array([1, -1])
    mus_true = np.vstack((mu1, mu2, mu3))

    cov1 = 0.1 * np.array([[1, 0], [0, 1]])
    cov2 = 0.4 * np.array([[1, 0], [0, 1]])
    cov3 = 0.2 * np.array([[1, 0], [0, 1]])
    cov_true = np.vstack((cov1, cov2, cov3))
    
    # Generate data draws from the gaussians
    x = np.linspace(-1,1,100)
    data1 = np.random.multivariate_normal(mu1, cov1, size=x.shape)
    data2 = np.random.multivariate_normal(mu2, cov2, size=x.shape)
    data3 = np.random.multivariate_normal(mu3, cov3, size=x.shape)

    # Combine into single dataset for algorithm
    data = np.vstack((data1, data2, data3)) 

    # Define useful data dimensions
    K = 3
    M = data.shape[1]
    N = data.shape[0]

    # Fabricate guesses for algorithm starting point
    # Should use K-Means to accomplish a better guess without prior knowledege
    # Mus are stored in K x M matrix
    mus_old = np.array([[-0.3, -0.3], [0.3, 0.3], [0.3, -0.3]])
    
    # Fabricate covariance starting point - stored in (M*K) x M matrix
    beta2 = 0.2
    cov_old = beta2 * np.eye((M))
    cov_old = np.vstack((cov_old, cov_old, cov_old))
    
    # Pis are stored in K-length array
    pis_old = np.array([0.33, 0.33, 0.34])

    # Initialize parameters with zero - these will be overwritten immeditately
    mus_new = np.zeros((K, M))
    cov_new = np.zeros((M, M))
    cov_new = np.vstack((cov_new, cov_new, cov_new))
    pis_new = [0, 0, 0]

    # Print starting point to console
    print('Initial guess:\nMeans: \n{}\n Sigmas: \n{}\n'\
        .format(mus_old, cov_old))

    # Initialize plotter variables needed in the loop
    data_list = [data1, data2, data3]
    colors = ['r', 'b', 'g']
    colors2 = ['tomato', 'slateblue', 'limegreen']
    
    # Initialize loop variables
    thresh = 0.01
    counter = 0
    ll_prev = 0
    ll_diff = 10
    while ll_diff > thresh: # Checks that the difference in the log-likelihood approaches zero

        # Make a movie! Plot dataset with ellipse for newly calculated parameters
        fig, ax_nstd = plt.subplots()
        ax_nstd.set_title('2D Expectation Maximization')
        ax_nstd.set_xlabel('x')
        ax_nstd.set_ylabel('y') 

        k = 0
        ax_nstd.scatter(data1[:,0], data1[:,1], color = colors[k])
        plot_ellipse(ax_nstd, mus_old[k, :], cov_old[k*2:k*2+M,:], color=colors2[k])

        k = 1
        ax_nstd.scatter(data2[:,0], data2[:,1], color = colors[k])
        plot_ellipse(ax_nstd, mus_old[k, :], cov_old[k*2:k*2+M,:], color=colors2[k])
        
        k = 2
        ax_nstd.scatter(data3[:,0], data3[:,1], color = colors[k])
        plot_ellipse(ax_nstd, mus_old[k, :], cov_old[k*2:k*2+M,:], color=colors2[k])

        plt.ion()
        plt.show(block=False)
        plt.pause(0.2)
        plt.close()
        
        for k in range(K):
            # Initialize counters for summed components 
            mu_count = np.zeros((M))
            sig_count = np.eye((M))
            N_k_count = 0
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
                sig_count += resp * np.matmul((x.reshape(-1, 1) - mus_new[k, :].reshape(-1, 1)), \
                    (x.reshape(-1, 1) - (mus_new[k, :].reshape(-1, 1))).T)
            
            # Eq. 9.28 - covariance formulation
            cov_new = (1 / N_k_count) * sig_count
            if k == 0:
                cov_new = (1 / N_k_count) * sig_count
            else:
                cov_stack = (1 / N_k_count) * sig_count
                cov_new = np.vstack((cov_new, cov_stack))
            if k > 0:
                cov_stack = (1 / N_k_count) * sig_count
                cov_new = np.vstack((cov_new, cov_stack))

        # Evaluate the log likelihood - Eq. 9.28
        # Use to determine when EM cannot find a better solution
        ll = 0
        for n in range(N):
            ll_count = 0
            x = data[n, :]
            for k in range(K):
                ll_count += pis_old[k] * multivariate_normal.pdf(x, mus_old[k, :], cov_old[k*2:k*2+M,:])
            ll += np.log(ll_count)
        ll_diff = np.abs(ll - ll_prev)
        ll_prev = ll

        # Re-assign parameters for next loop iteration
        mus_old = mus_new; cov_old = cov_new; pis_old = pis_new
        counter += 1

    print('Final values!\nActual Means: \n{}\nEM Means: \n{}\nActual Sigmas: \n{}\nEM Sigmas: \n{}\
        \nafter {} iterations'.format(mus_true, mus_new, cov_true, cov_new, counter))

    # Plot final values and maintain plot
    fig, ax_nstd = plt.subplots()
    ax_nstd.set_title('2D Expectation Maximization')
    ax_nstd.set_xlabel('x')
    ax_nstd.set_ylabel('y') 

    k = 0
    ax_nstd.scatter(data1[:,0], data1[:,1], color = colors[k])
    plot_ellipse(ax_nstd, mus_new[k, :], cov_new[k*2:k*2+M,:], color=colors2[k])

    k = 1
    ax_nstd.scatter(data2[:,0], data2[:,1], color = colors[k])
    plot_ellipse(ax_nstd, mus_new[k, :], cov_new[k*2:k*2+M,:], color=colors2[k])
    
    k = 2
    ax_nstd.scatter(data3[:,0], data3[:,1], color = colors[k])
    plot_ellipse(ax_nstd, mus_new[k, :], cov_new[k*2:k*2+M,:], color=colors2[k])

    plt.show(block=True)