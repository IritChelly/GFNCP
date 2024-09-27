import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import relabel
from plot_functions import plot_samples_RGB, plot_samples_Gauss2D, plot_samples_BW
import tikzplotlib as tkz


def data_invariance_metric(data_generator, dpmm, N=20, perms=500, Z=1000):
    '''
        Here we compute the probs of a given (gt) assignment for different (same data) permutations, 
        and for each trial we get prob_std_avg. 
        Then we plot all values in an histogram.
    '''
    
    print('Start computing data-invariance metric.')
    std_all = np.zeros(Z)
    
    for z in range(Z):
        print('z', z)
        probs = np.zeros(perms) # Stores the probabilities of getting the cs assignmnet, given each data-order permutation
        
        data, cs, clusters, K, _ = data_generator.generate(N=N, batch_size=1, train=False)    
        cs = cs.detach().cpu().numpy()
        for p in range(perms):
            arr = np.arange(N)  # Draw a permutation:
            np.random.shuffle(arr)   
            data_perm = data[:, arr, :] # permute the order of the data and the cs
            cs_perm = cs[0, arr]
            
            # Compute the probability of getting cs assignment with the current data-order permutation:
            probs[p] = compute_prob(dpmm, data_perm, cs_perm)
            
        prob_var = probs.var()
        prob_std = np.std(probs)
        prob_std_avg = prob_std / np.mean(probs)
        std_all[z] = prob_std_avg
    
    # Build a histogram from std_all:
    plt.clf()
    plt.cla()
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.subplots_adjust(left=0.15,
                    bottom=0.1, 
                    right=0.94, 
                    top=0.94, 
                    wspace=0.5, 
                    hspace=0.3)
    
    counts, bins = np.histogram(std_all, bins=20) 
    print('counts:', counts)
    print('bins:', bins)
    ax.hist(bins[:-1], bins, weights=counts, edgecolor='black', color='lightblue')
    ax.set_title('Clustering Probabilities std for different data permutations', fontsize='25')
    ax.set_xlabel('Bin Number')
    ax.set_ylabel('std/mean(probs)')
    # ax.set_xlim(left=min_data_lim, right=max_data_lim)
    
    fig.savefig('data_invariance_hist.png')
    fig.savefig('data_invariance_hist.pdf', format='pdf', bbox_inches='tight')
    tkz.save('data_invariance_hist.tex') 
    
    return fig, plt
    
    
         
def histogram_for_data_perms(data_orig, dpmm, params, N = 20, perms = 100):
    '''
    data_orig: one small dataset with N images. Shape: [1, N, channels, img_sz, img_sz]. Tensor.
    dpmm: a trained model
    N: number of images to cluster
    perms: number of data-order permutations to check. (We compute the probability of the most likely assignment given different data orders).
    '''
    
    # !!!! PERFORM THIS ANALYSIS SEVERAL TIMES TO GET LESS NOISY STD !!!
    
    probs_for_histogram = np.zeros(perms + 1) # Stores the probabilities of getting the most likely assignmnet, given each data-order permutation
    
    S = 100  # 5000  number of samples
    if params['channels'] == 0:
        data = data_orig.repeat(S, 1, 1)  # e.g.: [S, N, 2]. This is only one data point repeated S times.
    elif params['channels'] == 1:
        data = data_orig.repeat(S, 1, 1, 1)  # e.g.: [S, N, 28, 28]. This is only one data point repeated S times.
    elif params['channels'] == 3:
        data = data_orig.repeat(S, 1, 1, 1, 1)  # e.g.: [S, N, 28, 28, 3]. This is only one data point repeated S times.
    
    # Find the most likely clustering of "data":
    css, probs, _, _ = dpmm.module.sample(data, take_max=False)  # css: [M, N]; probs: [M,], where M is the number of succeeded samples.
    most_likely_clstr = css[np.argmax(probs), :]  # [N,]
    prob_most_likely_clstr = np.max(probs)  # scalar
        
    # Sanity check: compute the probability of getting "most_likely_clstr" using the original data order (no permutation):
    prob_orig_order = compute_prob(dpmm, data, most_likely_clstr)
    # print('\n(histogram) Probability of most-likely assignment, computation vs. sampler:', prob_orig_order, prob_most_likely_clstr, '\n')
    
    # Sample "perms" permutations of data and store their probability result from the model:    
    data = data[0, :][None, :]  # take the first row, as all rows are equal.
    for p in range(perms + 1):
        # Draw a permutation:
        arr = np.arange(N)
        np.random.shuffle(arr)   
        data_perm = data[:, arr, :] # permute the order of the data and the most likely assignment
        most_likely_clstr_perm = most_likely_clstr[arr]
        
        # Compute the probability of getting "most_likely_clstr_perm" with this data-order permutation:
        probs_for_histogram[p] = compute_prob(dpmm, data_perm, most_likely_clstr_perm)
    
    probs_for_histogram[perms] = prob_orig_order

    # Compute histogram variance:
    counts, bins = np.histogram(probs_for_histogram, bins=20, range=(0, 1))
    # print(counts, bins)
    fig = plt.figure(3, figsize=(15, 8))
    plt.clf()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel("Probabilities")
    plt.xlabel("Bin Number")
    plt.title('Histogram of the Most-likely Clustering Probabilities', fontsize='25')
    plt.xlim([0, 1])
    
    # plt.hist(probs_for_histogram, 20, histtype = 'bar', facecolor = 'blue', rwidth=2)

    prob_variance = probs_for_histogram.var()
    prob_std = np.std(probs_for_histogram)
    prob_std_avg = prob_std / np.mean(probs_for_histogram)
    
    return fig, plt, most_likely_clstr, prob_most_likely_clstr, prob_variance, prob_std, prob_std_avg


def compute_prob(dpmm, data, cs):
    # data: [1, data_dim]
    
    # Prepare cs:
    cs = relabel(cs)
    cs = torch.tensor(cs)[None, :]
    
    _, logprob_sum, _, _, _ = dpmm(data, cs)  # Here logprob_sum is B/(num_of_gpus) but we only need the first row.

    prob_sum = np.exp(- logprob_sum[0].cpu().numpy())
    return prob_sum
    

def plot_best_clustering(params, dataname, data, css, prob_css):
    '''
    data: one small dataset with N images in the original order. e.g if mnist the shape is: [1, N, 28, 28] 
    css: the most likely clustering for data. [N,]
    prob_css: probability of the most likely clustering. Scalar.
    '''
    # Plot the most-likely clustering and the original data order
    
    if params['channels'] == 0:
        data = data.clone().detach().to('cpu').numpy()
        fig, plt = plot_samples_Gauss2D(data, css, prob_css, rows=1, N=20, save_name=None)
    if params['channels'] == 1:
        fig, plt = plot_samples_BW(data, css, prob_css, rows=1, N=20, save_name=None)
    elif params['channels'] == 3:
        fig, plt = plot_samples_RGB(params, data, css, prob_css, rows=1, N=20, save_name=None)

    return fig, plt
