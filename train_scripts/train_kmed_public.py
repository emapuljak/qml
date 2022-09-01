import argparse
import numpy as np
import math
import h5py
import sys
sys.path.append('../')
sys.path.append('../../')
import qibo
qibo.set_backend("tensorflow")

import scripts.kmedians as KMed
import utils as u

def train_kmedians(latent_dim, train_size, read_file, seed=None, k=2, tolerance=1.e-3, save_dir=None):

    # read train data
    with h5py.File(read_file, 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]

        data_train = np.vstack([l1[:train_size], l2[:train_size]])
        if seed: np.random.seed(seed) # matters for small data sizes
        np.random.shuffle(data_train)

    kmedians = KMed.Kmedians(k=k, tolerance=tolerance)
    kmedians.fit(data_train)

    loss = kmedians.loss
    centroids = kmedians.centroids

    if save_dir:
        np.save(f'{save_dir}/centroids/final/centroids_lat{latent_dim}_{str(train_size)}.npy', centroids)
        np.save(f'{save_dir}/loss/final/LOSS_lat{latent_dim}_{str(train_size)}.npy', loss)
        print('Centroids and labels saved!')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='read arguments for qkmedians training')
    parser.add_argument('-latent_dim', dest='latent_dim', type=int, help='latent dimension')
    parser.add_argument('-train_size', dest='train_size', type=int, help='training data size')
    parser.add_argument('-read_file', dest='read_file', type=str, help='training data file')
    parser.add_argument('-seed', dest='seed', type=int, help='seed for consistent results')
    parser.add_argument('-k', dest='k', type=int, default=2, help='number of classes')
    parser.add_argument('-tolerance', dest='tolerance', type=float, default=1.e-3, help='tolerance')
    parser.add_argument('-save_dir', dest='save_dir',type=str, help='directory to save results')

    args = parser.parse_args()
    
    # latent_dim=['8', '16', '24', '32', '40']
    # for j in range(len(latent_dims)):
    #     print(f'========== LATENT DIM {latent_dims[j]} ==========')
    #     read_file = f'/eos/user/e/epuljak/private/epuljak/public/AE_data/latent/lat{latent_dims[j]}/latentrep_QCD_sig.h5'
    #     save_dir = '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_kmedians/diJet'
    #     train_kmedians(latent_dims[j], args.train_size, read_file, args.seed, args.k, args.tolerance, save_dir)
    
    train_size = [10, 6000]
    for j in range(len(train_size)):
        print(f'========== Train_size {train_size[j]} ==========')
        read_file = f'/eos/user/e/epuljak/private/epuljak/public/AE_data/latent/lat{args.latent_dim}/latentrep_QCD_sig.h5'
        save_dir = '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_kmedians/diJet'
        train_kmedians(args.latent_dim, train_size[j], read_file, args.seed, args.k, args.tolerance, save_dir)
    