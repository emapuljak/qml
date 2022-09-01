import argparse
import numpy as np
import math
import h5py
import sys
sys.path.append('../')
sys.path.append('../../')
import qibo
qibo.set_backend("tensorflow")

import scripts.qkmedians as qkmed
import utils as u

def train_qkmedians(latent_dim, train_size, read_file, device_name, seed=None, k=2, tolerance=1.e-3, save_dir=None):

    # read train data
    with h5py.File(read_file, 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]

        data_train = np.vstack([l1[:train_size], l2[:train_size]])
        if seed: np.random.seed(seed) # matters for small data sizes
        np.random.shuffle(data_train)

    centroids = qkmed.initialize_centroids(data_train, k)   # Intialize centroids

    i = 0; new_tol=1
    loss=[]
    while True:
        cluster_label, _ = qkmed.find_nearest_neighbour_DI(data_train,centroids, device_name)       # find nearest centroids
        print(f'Found cluster assignments for iteration: {i+1}')
        new_centroids = qkmed.find_centroids_GM(data_train, cluster_label, clusters=k) # find centroids

        loss_epoch = np.linalg.norm(centroids - new_centroids)
        loss.append(loss_epoch)
        if loss_epoch < tolerance:
            centroids = new_centroids
            print(f"Converged after {i+1} iterations.")
            break
        elif loss_epoch > tolerance and i > new_tol*200:     # if after 200*new_tol epochs, difference != 0, lower the tolerance
            print(f'Rising the tolerance after {i}th iteration')
            tolerance *= 10
            new_tol += 1
        i += 1     
        centroids = new_centroids

    if save_dir:
        np.save(f'{save_dir}/cluster_labels/final/cluster_label_lat{latent_dim}_{str(train_size)}.npy', cluster_label)
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
    parser.add_argument('-device_name', dest='device_name',type=str, help='name of GPU if exists')

    args = parser.parse_args()
    
    if args.device_name: qibo.set_device(args.device_name)
    
    # latent_dims=['4', '8', '16', '24', '32', '40']
    # for j in range(len(latent_dims)):
    #     print(f'========== LATENT DIM {latent_dims[j]} ==========')
    #     read_file = f'/eos/user/e/epuljak/private/epuljak/public/AE_data/latent/lat{latent_dims[j]}/latentrep_QCD_sig.h5'
    #     save_dir = '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/diJet'
    #     train_qkmedians(latent_dims[j], args.train_size, read_file, args.device_name, args.seed, args.k, args.tolerance, save_dir)
    
    train_size = [10, 6000]
    for j in range(len(train_size)):
        print(f'========== Train_size {train_size[j]} ==========')
        read_file = f'/eos/user/e/epuljak/private/epuljak/public/AE_data/latent/lat{args.latent_dim}/latentrep_QCD_sig.h5'
        save_dir = '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/diJet'
        train_qkmedians(args.latent_dim, train_size[j], read_file, args.device_name, args.seed, args.k, args.tolerance, save_dir)