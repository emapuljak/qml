from qibo import K
from qibo.config import raise_error
from qibo.core import measurements
from qibo.abstractions.states import AbstractState
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import h5py
import math
import sys
sys.path.append('../')
sys.path.append('../../')
import scripts.kmedians as KMed

def symbolicVectorState(state_vector, decimals=5, cutoff=1e-10, max_terms=20):

    """Dirac notation representation of the state in the computational basis.
        Args:
            decimals (int): Number of decimals for the amplitudes.
                Default is 5.
            cutoff (float): Amplitudes with absolute value smaller than the
                cutoff are ignored from the representation.
                Default is 1e-10.
            max_terms (int): Maximum number of terms to print. If the state
                contains more terms they will be ignored.
                Default is 20.
        Returns:
            A string representing the state in the computational basis.
        """
    
    state = state_vector.numpy()
    terms = []
    for i in K.np.nonzero(state)[0]:
        b = bin(i)[2:].zfill(state_vector.nqubits)
        if K.np.abs(state[i]) >= cutoff:
            x = np.round(state[i], decimals)
            terms.append(f"{x}|{b}>")
        if len(terms) >= max_terms:
            terms.append("...")
            break
    return " + ".join(terms)

def symbolicMatrixState(state_matrix, decimals=5, cutoff=1e-10, max_terms=20):
    state = state_matrix.numpy()
    terms = []
    indi, indj = K.np.nonzero(state)
    for i, j in zip(indi, indj):
        bi = bin(i)[2:].zfill(state_matrix.nqubits)
        bj = bin(j)[2:].zfill(state_matrix.nqubits)
        if K.np.abs(state[i, j]) >= cutoff:
            x = round(state[i, j], decimals)
            terms.append(f"{x}|{bi}><{bj}|")
        if len(terms) >= max_terms:
            terms.append("...")
            break
    return " + ".join(terms)

def load_data_and_centroids_c(run, n_samples_train=500, n_samples_test=400, qcd_test_size=500, k=2, signal_name='RSGraviton_WW_NA', mass='3.5', br_na=None):

    # read QCD predicted data (test - SIDE)
    read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/{run}/'
    file_name = 'latentrep_QCD_sig.h5'
    with h5py.File(read_dir+file_name, 'r') as file:
        data = np.array(file['latent_space'][:])
        data_for_c = data[:n_samples_train]
        data = data[-qcd_test_size:]

    # read SIGNAL predicted data
    read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/{run}/'
    file_name = f'latentrep_{signal_name}.h5'
    with h5py.File(read_dir+file_name, 'r') as file:
        #data_s = file['latent_space_NA_RSGraviton_WW_NA_3.5'][:n_samples_test]
        if br_na:
            data_s = file[f'latent_space_{br_na}_{signal_name}_{mass}'][:n_samples_test]
        else: data_s = file[f'latent_space_{signal_name}_{mass}'][:n_samples_test]
    
    kmedians = KMed.Kmedians(k=k)
    kmedians.fit(data_for_c)

    centroids_c = kmedians.medians
    return data, data_s, centroids_c
        
        
def get_metric(qcd, bsm, tpr_window=[0.5, 0.6]):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)

    # AUC error
    n_n = qcd.shape[0]
    n_p = bsm.shape[0]
    D_p = (n_p - 1) * ((auc_data/(2 - auc_data)) - auc_data**2)
    D_n = (n_n - 1) * ((2 * auc_data**2)/(1 + auc_data) - auc_data**2)
    auc_error = np.sqrt((auc_data * (1 - auc_data) + D_p + D_n)/(n_p * n_n))

    # FPR and its error
    position = np.where((tpr_loss>=tpr_window[0]) & (tpr_loss<=tpr_window[1]))[0][0]
    threshold_data = threshold_loss[position]
    pred_data = [1 if i>= threshold_data else 0 for i in list(pred_val)]
    tn, fp, fn, tp = confusion_matrix(true_val, pred_data).ravel()
    fpr_data = fp / (fp + tn)
    print(f'FPR: {fpr_data}')
    #one_over_fpr = 1./fpr_data
    print(f'TN+FP: {(tn + fp)}')
    fpr_error = np.sqrt(fpr_data * (1 - fpr_data) / (tn + fp))
    print(f'FPR ERROR: {fpr_error}')
    #one_over_fpr_error = 1./fpr_error
    #print(tpr_error)
    return (auc_data, auc_error), (fpr_data, fpr_error)