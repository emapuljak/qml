from qibo import K
from qibo.config import raise_error
from qibo.core import measurements
from qibo.abstractions.states import AbstractState
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target
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

def find_pT_jet_of_min(dijet_features, indices):
    # there are only 2 jets here
    pT_array_1 = dijet_features[:,1]; pT_array_2 = dijet_features[:,6] #j1pt, j2pt
    pTs = []
    for j, i in enumerate(indices):
        if i==0: pTs.append(pT_array_1[j])
        elif i==1: pTs.append(pT_array_2[j])
    return np.array(pTs)

def find_pT_phi_particles_of_min(pT_array, phi_array, indices):
    # there are only 2 jets here
    pT_array_1 = pT_array[0]; pT_array_2 = pT_array[1]
    pTs = []
    for j, i in enumerate(indices):
        if i==0: pTs.append(pT_array_1[j,:])
        elif i==1: pTs.append(pT_array_2[j,:])
    
    phi_array_1 = phi_array[0]; phi_array_2 = phi_array[1]
    phis = []
    for j, i in enumerate(indices):
        if i==0: phis.append(phi_array_1[j,:])
        elif i==1: phis.append(phi_array_2[j,:])
    return np.array(pTs), np.array(phis)

def calc_pT_jet(pT_particles, phi_particles):
    # px = pT*cos(phi)
    px_particles = pT_particles*np.cos(phi_particles)
    # py = pT*sin(phi)
    py_particles = pT_particles*np.sin(phi_particles)
    
    px_Jet = np.sum(px_particles, axis=1)
    py_Jet = np.sum(py_particles, axis=1)
    
    #pT_Jet = sqrt(px_jet^2+py_jet^2)
    pT_Jet = np.sqrt(np.power(px_Jet, 2) + np.power(py_Jet, 2))
    return pT_Jet

def combine_loss_min_index(loss_j1, loss_j2):
    index_min = []
    for l1, l2 in zip(loss_j1, loss_j2):
        index_min.append(np.argmin([l1, l2]))
    return np.array(index_min)

def combine_loss_min(loss):
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2), combine_loss_min_index(loss_j1, loss_j2)

def pearson_coef(data1, data2):
    #covariance = np.cov(data1, data2)
    #return covariance / (np.std(data1) * np.std(data2))
    corr, _ = pearsonr(data1, data2)
    return corr
    
def load_data_and_centroids_c(run, i=1, n_samples_train=500, n_samples_test=400, qcd_test_size=500, k=2, signal_name='RSGraviton_WW_NA', mass='3.5', br_na=None, around_peak=None, centroids_c_dir=None, loss_c_dir=None):

    # read QCD predicted data (test - SIDE)
    read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/corrected_cuts/{run}/'
    file_name = 'latentrep_QCD_sig.h5'
    with h5py.File(read_dir+file_name, 'r') as file:
        data = np.array(file['latent_space'][:])
        data_for_c = data[n_samples_train*i:n_samples_train*(i+1)]
        data = data[-qcd_test_size:]
        
    # read SIGNAL predicted data
    read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/corrected_cuts/{run}/'
    if around_peak: file_name = f'latentrep_{signal_name}_{around_peak}.h5'
    else: file_name = f'latentrep_{signal_name}.h5'
    with h5py.File(read_dir+file_name, 'r') as file:
        #data_s = file['latent_space_NA_RSGraviton_WW_NA_3.5'][:n_samples_test]
        if br_na:
            data_s = file[f'latent_space_{br_na}_{signal_name}_{mass}'][:n_samples_test]
        else: data_s = file[f'latent_space_{signal_name}_{mass}'][:n_samples_test]
    
    if centroids_c_dir: 
        centroids_c = np.load(centroids_c_dir)
        loss = np.load(loss_c_dir)
    else:
        kmedians = KMed.Kmedians(k=k)
        kmedians.fit(data_for_c)

        loss = kmedians.loss
        centroids_c = kmedians.medians

    
    return data, data_s, centroids_c, loss

def load_clustering_test_data_iML(lat_dim, test_size=500, k=2, signal_name='RSGraviton_WW', mass='35', br_na=None, around_peak=None, read_dir='/eos/user/k/kiwoznia/data/laspaclu_results/latent_rep/ae_run_50'):

    # read QCD latent space data
    file_name = f'{read_dir}/qcdSigExt.h5'
    with h5py.File(file_name, 'r') as file:
        data = file['latent_ae']
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        
        r_index = np.random.choice(list(range(l1.shape[0])), size=int(test_size/2))
        data_test_qcd = np.vstack([l1[r_index], l2[r_index]])
        
    # read SIGNAL predicted data
    file_name = f'{read_dir}/GtoWW35na.h5'
    with h5py.File(file_name, 'r') as file:
        data = file['latent_ae']
        print(data.shape)
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        
        r_index = np.random.choice(list(range(l1.shape[0])), size=int(test_size/2))
        data_test_sig = np.vstack([l1[r_index], l2[r_index]])
    
    return data_test_qcd, data_test_sig

def load_clustering_test_data(lat_dim, test_size=10000, k=2, signal_name='RSGraviton_WW', mass='35', br_na=None, around_peak=None, read_dir='/eos/user/e/epuljak/private/epuljak/public/diJet', split=False, n_folds=None):

    # read QCD latent space data
    file_name = f'{read_dir}/lat{lat_dim}/latentrep_QCD_sig_testclustering.h5'
    with h5py.File(file_name, 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        
        if split:
            l1 = l1[:test_size]; l2 = l2[:test_size]
            l1_split = np.split(l1, n_folds, axis=0)
            l2_split = np.split(l2, n_folds, axis=0)
            test_size_fold = math.floor(test_size/n_folds)
            splited_data_test_qcd=[]
            for i in range(n_folds):
                data_fold = np.vstack([l1_split[i], l2_split[i]])
                print(data_fold.shape)
                splited_data_test_qcd.append(data_fold)
        else:
            #r_index = np.random.choice(list(range(l1.shape[0])), size=int(test_size/2))
            data_test_qcd = np.vstack([l1[:test_size], l2[:test_size]])
            print(data_test_qcd.shape)
    
    # read SIGNAL predicted data
    read_dir =f'{read_dir}/lat{lat_dim}'
    if br_na:
        signal = f'{signal_name}_{br_na}_{mass}'
    else: signal=f'{signal_name}_{mass}'
    if around_peak:
        print(around_peak)
        file_name = f'{read_dir}/latentrep_{signal}_{around_peak}.h5'
    else: file_name = f'{read_dir}/latentrep_{signal}.h5'
    with h5py.File(file_name, 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        if split:
            l1 = l1[:test_size]; l2 = l2[:test_size]
            l1_split = np.split(l1, n_folds, axis=0)
            l2_split = np.split(l2, n_folds, axis=0)
            test_size_fold = math.floor(test_size/n_folds)
            splited_data_test_sig=[]
            for i in range(n_folds):
                data_fold = np.vstack([l1[:test_size_fold], l2[:test_size_fold]])
                print(data_fold.shape)
                splited_data_test_sig.append(data_fold)
        else:
            #r_index = np.random.choice(list(range(l1.shape[0])), size=int(test_size/2))
            data_test_sig = np.vstack([l1[:test_size], l2[:test_size]])
            print(data_test_sig.shape)
    
    if split: return splited_data_test_qcd, splited_data_test_sig
    return data_test_qcd, data_test_sig
   
def ad_score(cluster_assignments, distances, method='sum_all'):
    if method=='sum_all':
        return np.sqrt(np.sum(distances**2, axis=1))
    else:
        return np.sqrt(distances[range(len(distances)), cluster_assignments]**2)
    
def get_auc(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)
    
    return auc_data
        
def get_metric(qcd, bsm, tpr_window=[0.5, 0.6]):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    # auc_data = auc(fpr_loss, tpr_loss)

    # AUC error
    # n_n = qcd.shape[0]
    # n_p = bsm.shape[0]
    # D_p = (n_p - 1) * ((auc_data/(2 - auc_data)) - auc_data**2)
    # D_n = (n_n - 1) * ((2 * auc_data**2)/(1 + auc_data) - auc_data**2)
    # auc_error = np.sqrt((auc_data * (1 - auc_data) + D_p + D_n)/(n_p * n_n))

    # FPR and its error
    position = np.where((tpr_loss>=tpr_window[0]) & (tpr_loss<=tpr_window[1]))[0][0]
    threshold_data = threshold_loss[position]
    pred_data = [1 if i>= threshold_data else 0 for i in list(pred_val)]
    tn, fp, fn, tp = confusion_matrix(true_val, pred_data).ravel()
    fpr_data = fp / (fp + tn)
    print(f'FPR: {fpr_data}')
    one_over_fpr_data = 1./fpr_data # y = 1/x
    print(f'TN+FP: {(tn + fp)}')
    fpr_error = np.sqrt(fpr_data * (1 - fpr_data) / (fp + tn))
    one_over_fpr_error = fpr_error*(1./np.power(fpr_data,2)) # sigma_y = sigma_x * (1/x^2)
    print(f'FPR ERROR: {fpr_error}')
    print(f'1/FPR error: {one_over_fpr_error}')
    #print(tpr_error)
    #return (auc_data, auc_error), (one_over_fpr_data, one_over_fpr_error)
    return one_over_fpr_data, one_over_fpr_error


def make_data_dist_plots(feature_data, feature_data2, xlabel, bins, density, title, color='blue', linewidth=2, ranges=None, ylimit=None, xlimit=None):
    plt.figure(figsize=(9,7))
    #if ranges == None: ranges = ut.find_min_max_range(true, prediction) 
    plt.hist(feature_data, bins=bins, histtype='step', density=density, color=color, linewidth=linewidth)
    plt.hist(feature_data2, bins=bins, histtype='step', density=density, color='red', linewidth=linewidth)
    plt.yscale('log', nonpositive='clip')
    plt.ylabel('Prob. Density(a.u.)')
    plt.xlabel(xlabel)
    plt.tight_layout()
    if ylimit != None: plt.ylim(ylimit)
    if xlimit != None: plt.xlim(xlimit)
    #plt.legend([])
    plt.title(title)
    plt.show()
    
    
def get_thresholds(y_true, y_score,  pos_label=None):
    # Check to make sure y_true is valid
    # y_type = type_of_target(y_true, input_name="y_true")
    # if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
    #     raise ValueError("{0} format is not supported".format(y_type))

    #check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)


    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    thresholds = y_score[threshold_idxs]
    #thresholds = np.r_[thresholds[0] + 1, thresholds] # 
    return thresholds

def get_roc_data(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val, drop_intermediate=False)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data

def get_roc_data_byhand(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    
    thresholds = u.get_thresholds(true_val, pred_val)
        
    fpr = []; tpr = []

    for threshold in thresholds:

        y_pred = np.where(pred_val >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (true_val == 0))
        tp = np.sum((y_pred == 1) & (true_val == 1))

        fn = np.sum((y_pred == 0) & (true_val == 1))
        tn = np.sum((y_pred == 0) & (true_val == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
        
    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[thresholds[0] + 1, thresholds]
    return fpr, tpr, thresholds