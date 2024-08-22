#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["NCCL_P2P_DISABLE"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_SILENT"] = "true"

import numpy as np
import argparse
import time
# import tensorflow as tf
import torch
from ncp import NeuralClustering
from data_generator import get_generator
from utils import *
from params import get_parameters
from losses import kl_loss_func, j_loss_func, mc_loss_func
from evaluation import eval_stats, eval_stats_full_test, plot_samples_and_histogram
import shutil
from collections import OrderedDict
import random

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
    

def main(args):

    datasetname = args.dataset
    params = get_parameters(datasetname)
    params['device'] = torch.device("cuda:0" if args.cuda else "cpu") 
    params['dataset_name'] = datasetname
    
    seed = args.seed
    set_seed(seed)

    wnb = init_wandb(args, params)
    
    batch_size = params['batch_size']
    loss_str = params['loss_str']
    max_it = params['max_it']
    epochs = 1
    num_sched_steps_per_epoch = max_it // params['sched_lr_update']
    lr = params['lr']
    min_lr = params['min_lr']
    weight_decay = params['weight_decay']
    weight_decay_end = params['weight_decay_end']
    device = params['device']
    N_sampling = params['N_sampling']
    show_histogram = args.show_histogram  # A flag for analyzing a trained model (histogram)
    lambda_mc = params['lambda_mc']
    lambda_j = params['lambda_j']
    lambda_entrpy = params['lambda_entrpy']
    mc_weights = params['mc_weights']
    j_weights = params['j_weights']
    plot_freq = params['plot_freq']
    
    # Define the model:
    dpmm = NeuralClustering(params).to(params['device'])
    
    # Define the data generator:
    data_generator = get_generator(params)
    
    # Define learning rate and optimizers:
    optimizer = torch.optim.Adam(dpmm.parameters() , lr=lr, weight_decay=weight_decay)
    
    # Define lr and weight decay schedulers:
    print("Use Cosine LR scheduler")
    lr_schedule_values = cosine_scheduler(
        lr, min_lr, epochs, num_sched_steps_per_epoch,
        warmup_epochs=0, warmup_steps=0,)
    
    print('lr_schedule_values size:', len(lr_schedule_values))
    
    if weight_decay_end is None:
        weight_decay_end = weight_decay
    
    wd_schedule_values = cosine_scheduler(
        weight_decay, weight_decay_end, epochs, num_sched_steps_per_epoch)
    
    it = 0
    it_lr_sched = 0
    it_lr_sched_ncp = 0
    
    # Object that stores the model info for saving:
    state = dict(optimizer=optimizer, model=dpmm, step=0)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join('saved_models/', datasetname, 'checkpoints')
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join('saved_models/', datasetname, 'checkpoints-meta', 'checkpoint.pth')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # tf.io.gfile.makedirs(checkpoint_dir)
    # tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))

    # Load the trained model if required:
    state = restore_checkpoint(checkpoint_meta_dir, state, params['device'])
    print('\nRestore model from iteration:', state['step'])
    print('Learning rate:', params['lr'])
    
    
    # ----------------------------------------------
    #              Main training loop:
    # ----------------------------------------------
    
    # data_0, cs_0, clusters_0, K_0 = data_generator.generate(None, batch_size)    
    # N = data_0.shape[1]
    
    stats = {'NMI_sum': 0, 'ARI_sum': 0, 'LL_sum': 0, 'count': 0, 
             'NMI_max': 0, 'ARI_max': 0, 'LL_max': 0, 'NMI_max_it': 0, 'ARI_max_it': 0, 'LL_max_it': 0, 
             'NMI_full_test_max': 0, 'ARI_full_test_max': 0, 'NMI_full_test_max_it': 0, 'ARI_full_test_max_it': 0}
    
    while it < max_it:
                
        dpmm.train()  # new model
        
        # Update learning rate & weight decay:
        if (lr_schedule_values is not None or wd_schedule_values is not None) and it % params['sched_lr_update'] == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it_lr_sched]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it_lr_sched]
            
            it_lr_sched = it_lr_sched + 1
            
        # Evaluate the model periodically:
        if it % plot_freq == 0 or it == 1:
            print('\nPloting samples, compute NMI and test loss, iteration ' + str(it) + '.. \n')   
            
            # NMI, ARI, LL.
            data, cs_gt, clusters, K = data_generator.generate(N=N_sampling, batch_size=1, train=False)  # data: [1, N, 2] or [1, N_sampling, 28, 28] or [1, N, 3, 28, 28]
            stats = eval_stats(wnb, data, cs_gt, params, dpmm, it, stats)
            
            # Plots. Here we must use N=20 because we need to plot the results:
            data, cs_gt, clusters, K = data_generator.generate(N=20, batch_size=1, train=False)  # data: [1, N, 2] or [1, N, 28, 28] or [1, N, 3, 28, 28]            
            plot_samples_and_histogram(wnb, data, cs_gt, params, dpmm, it, N=20, show_histogram=show_histogram)

            # # Test the model on the whole tests data:
            # data, cs_gt = data_generator.get_full_test_data() 
            # stats = eval_stats_full_test(wnb, data, cs_gt, params, dpmm, it, stats)
            
            
        # Save the model periodically:
        if it % 100 == 0:
            print('\Saving model.. \n') 
            save_model(state, it, dpmm, optimizer, checkpoint_dir, checkpoint_meta_dir)
  
        # Generate one batch for training
        data, cs, clusters, K = data_generator.generate(None, batch_size, train=True)    
        cs = relabel(cs)    # this makes cluster labels appear in cs[] in increasing order
        N = data.shape[1]

        # Display adaptive learning rate
        curr_lr = optimizer.param_groups[0]['lr']
            
        print('Iteration: {0} N: {1} Current lr: {2:.10f}'.format(it, N, curr_lr))

        # Saves the cs values computed during training, for NMI computation
        cs_train = np.zeros((batch_size, N))
        
        # Training of one point: FW and Backprop of one batch.
        # (Each training step includes a few permutations of the data order)   
        dpmm.train()
        
        dpmm.encode(data)
        # dpmm_ncp.previous_n = 0   
        
        kl_loss = 0  
        mc_loss = 0
        log_pn = 0
        
        # FW step (includes N-1 FW steps in the net):
        for n in range(1, N):  # n is the next point to be assigned. Points up to n-1 (in each point in the batch, so there are B * (n-1)) were already assigned.                              
            E, E_mask = dpmm(cs, n)   # E is [B, K + 1]. This is the unnormalized logprob of p(c_{0:n} | x).
            
            # -------- KL Loss: --------------------
            # Get the logprobs which is log p(c_n|c_1..c_n-1, x)
            logprobs_kl = kl_loss_func(E, E_mask)
            c = cs[n] # The ground-truth cluster of the n-th point (which is similar in all B datasets)
            cs_train[:, n] = np.argmax(logprobs_kl.clone().detach().to('cpu').numpy(), axis=1)  # [B, 1] compute this for NMI computation later.
            kl_loss -= logprobs_kl[:, c].mean() # The loss is minus the value in the relevant cluster assignment in logprobs.
            # -------- END KL Loss: ----------------
            
            
            # # -------- J Loss: --------------------
            if n == N - 1:
                j_loss, data_E, entrpy = j_loss_func(dpmm, E, N, n, cs, it)  # scalar
            # # -------- END J Loss: ----------------
            
            
            # -------- MC Loss: --------------------
            mc_n_term, log_pn = mc_loss_func(E, E_mask, log_pn, n, N, cs, batch_size, it, device, epsilon=1e-5)
            mc_loss += mc_n_term
            # -------- END MC Loss: ----------------
            

        mc_loss = mc_loss / N

        # Choose the loss:
        if loss_str == 'MC + J':
            loss = lambda_j * j_loss + lambda_mc * mc_loss - lambda_entrpy * entrpy
        elif loss_str == 'MC + J + KL':
            loss = lambda_j * j_loss + lambda_mc * mc_loss + kl_loss - lambda_entrpy * entrpy
        elif loss_str == 'J':
            loss = lambda_j * j_loss - lambda_entrpy * entrpy
        elif loss_str == 'MC':
            loss = lambda_mc * mc_loss
        elif loss_str == 'KL':
            loss = kl_loss

        loss.backward()    # this accumulates the gradients for each permutation
        
        NMI_train = compute_NMI(cs, cs_train, None)
        ARI_train = compute_ARI(cs, cs_train, None)           
                
        # Store statistics in wandb:
        sts = update_stats_train(it, N, K, loss, kl_loss, mc_loss, j_loss, entrpy, NMI_train, ARI_train)  # stats.update({'train_acc1': acc_train})

        wandb.log(sts, step=it)
        
        optimizer.step()      # the gradients used in this step are the sum of the gradients for each permutation 
        optimizer.zero_grad()    
        
        it += 1
        
    
    # Print avg metrics:
    print('\n * Max NMI (test): {0:.3f} (on iter:) {1:.3f}, Max ARI (test): {2:.3f} (on iter:) {3:.3f}, Max LL (test): {4:.3f} (on iter:) {5:.3f}'.format(stats['NMI_max'], stats['NMI_max_it'], stats['ARI_max'], stats['ARI_max_it'])) 
    print('\n')
    
    

def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def init_wandb(args, params):
    if has_wandb:
        wnb = wandb.init(entity='bgu_cs_vil', project="NCP_EB", name=args.experiment, config=args)
        wnb.log_code(".")  # log source code of this run
        wnb.config.update(params)
    else:
        wnb = None
        print("Problem with initiating wandb.")
    
    return wnb
   
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Neural Clustering Process')

    parser.add_argument('--dataset', type=str, default='Gauss2D', metavar='S',
                    choices = ['Gauss2D','MNIST', 'FASHIONMNIST', 'CIFAR', 'Features'],
                    help='Generative Model: Gauss2D or MNIST (default: Gauss2D)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 123)')
    parser.add_argument('--show-histogram', action='store_true', default=False,
                    help='flag for analyzing a trained model')
    parser.add_argument('--load-model', action='store_true', default=False,
                    help='flag for loading model or start from scratch')       
    parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of wandb experiment')   
        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if not args.load_model:
        # Remove saved models
        model_dir = 'saved_models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        shutil.rmtree(model_dir)
    
    main(args)

