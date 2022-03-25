import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import h5py
from datetime import datetime
import qibo
qibo.set_backend("tensorflow")
import sys
sys.path.append('../')
sys.path.append('../../')
import scripts.qkmeans as qkm
import scripts.minimization as m
import scripts.qkmedians as qkmed
import scripts.oracle as o
import utils as u
import plots as p
import sys 

# stdoutOrigin=sys.stdout 
# sys.stdout = open("output.txt", "w")

qibo.set_device("/GPU:0")

run='04032022_1'

# read QCD predicted data (test - SIDE)
read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/{run}/'
file_name = 'latentrep_QCD_sig.h5'
with h5py.File(read_dir+file_name, 'r') as data:
    latent_rep = data['latent_space'][:4000]
    
# read SIGNAL predicted data
# read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/{run}/'
# file_name = 'latentrep_RSGraviton_WW_NA.h5'
# with h5py.File(read_dir+file_name, 'r') as data:
#     latent_rep_sig = data['latent_space_NA_RSGraviton_WW_NA_3.5'][:3200]
    
# data_s = latent_rep_sig # signal
data = latent_rep #qcd

k = 2 # number of clusters
centroids = qkmed.initialize_centroids(data, k)   # Intialize centroids

tolerance=1e-3

# run k-medians algorithm
i = 0
while True:
    cluster_label, _ = qkmed.find_nearest_neighbour_DI(data,centroids)       # find nearest centers
    print(f'Found cluster assignments for iteration: {i+1}')
    new_centroids = qkmed.find_centroids_GM(data,cluster_label, clusters=k)               # find centroids

    if np.linalg.norm(centroids - new_centroids) < tolerance:
        centroids = new_centroids
        print(f"Converged after {i+1} iterations.")
        break
    print(f'Iteration: {i+1}')
    i += 1     
    centroids = new_centroids
    if i == 150: break
print('QKmedians converged!')

np.save(f'cluster_label_{run}_Durr_DI_AE_4000.npy', cluster_label)
np.save(f'centroids_{run}_Durr_DI_AE_4000.npy', centroids)

print('Centroids and labels saved!')
#p.plot_latent_representations(data, cluster_label, '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/plots/latent_dim16', f'jets_qkmedians_{run}_500_durr_GM_AE' )
#print("Plotted latent representations!")
#p.plot_centroids(centroids, '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/plots/latent_dim16', f'jets_qkmedians_{run}_500_durr_GM_AE')
#print("Plotted centroids!")
#cluster_label_q, q_distances = qkmed.find_nearest_neighbour_DI(data, centroids)
#cluster_label_s, q_distances_s = qkmed.find_nearest_neighbour_DI(data_s,centroids)       # find nearest centers

# p.plot_latent_representations(data_s, cluster_label_s, '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/plots/latent_dim16', f'jets_qkmedians_{run}_500_durr_GM_AE_SIGNALdata')
# print("Plotted latent representations test piece!")

# sys.stdout.close()
# sys.stdout=stdoutOrigin
