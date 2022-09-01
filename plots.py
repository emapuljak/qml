import pandas as pd
#import mplhep
import time
import seaborn as sns
import pathlib
import h5py
import matplotlib.pyplot as plt
plt.rcParams['legend.title_fontsize'] = 'xx-small'
import utils as u
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve, auc
import mplhep as hep
plt.style.use(hep.style.CMS)
import sys
sys.path.append('../')
sys.path.append('../../')
import scripts.qkmedians as qkmed
import utils as u
import plots as p
import scripts.util as ut
import scripts.classic_functions as cf

def plot_latent_representations(data, class_labels, save_dir=None, sample_id=None):
    df = pd.DataFrame(data)
    df['class_id'] = class_labels
        
    figure = sns.pairplot(df, hue='class_id', diag_kind="hist")
    plt.tight_layout()
    if save_dir:
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        figure.savefig(save_dir+'/latent_feature_pairs_'+sample_id+'.png')
    plt.show()

def plot_centroids(centroids, save_dir, sample_id, clusters=2):
    
    k = centroids.shape[1]
    fig, ax = plt.subplots(k, k, sharex='col', sharey='row', figsize=(15,15))
    
    #rows, cols = np.tril_indices(8, m=8)
    for i in range(k):
        for j in range(k):
            if i<j:
                ax[i, j].axis('off')
            else:
                ax[i, j].scatter(centroids[0,i], centroids[0,j], c='red', s=20, marker="X") #cluster 0
                ax[i, j].scatter(centroids[1,i], centroids[1,j], c='blue', s=20, marker="D") #cluster 1
                #ax[i, j].scatter(centroids[2,i], centroids[2,j], c='green', s=50, marker="o") #cluster 1
                #ax[i, j].grid(True, fillstyle='full')
            
    fig.savefig(save_dir+'/centroids_'+sample_id+'.png')
    plt.show()
    
def plot_centroids_compare(centroids_q, centroids_c, fig_dir, sample_id, clusters=2):
    
    k = centroids_q.shape[1]
    fig, ax = plt.subplots(k, k, sharex='col', figsize=(20,20))
    #set_share_axes(, sharex=True)
    #set_share_axes(ax[:,2:], sharex=True)
    
    xs = np.linspace(-1,1,200)
    #rows, cols = np.tril_indices(8, m=8)
    for i in range(k):
        for j in range(k):
            if i==j:
                sns.kdeplot(centroids_q[:,i], ax=ax[i,j], fill=True, color='green')
                sns.kdeplot(centroids_c[:,i], ax=ax[i,j], fill=True, color='maroon')
                ax[i, j].grid(True, fillstyle='full')
                ax[i, j].set_xlim(-1,1)
                ax[i, j].set_yticklabels([])
                ax[i, j].set(ylabel=None)
                #ax[i, j].set_ylim(-1,1)
            else:
                ax[i, j].scatter(centroids_q[0,j], centroids_q[0,i], c='limegreen', s=50, marker='X') #cluster 0
                ax[i, j].scatter(centroids_q[1,j], centroids_q[1,i], c='green', s=50, marker='X') #cluster 1
                ax[i, j].scatter(centroids_q[2,j], centroids_q[2,i], c='forestgreen', s=50, marker='X') #cluster 1
                    
                ax[i, j].scatter(centroids_c[0,j], centroids_c[0,i], c='indianred', s=50, marker='D') #cluster 0
                ax[i, j].scatter(centroids_c[1,j], centroids_c[1,i], c='maroon', s=50, marker='D') #cluster 1
                ax[i, j].scatter(centroids_c[2,j], centroids_c[2,i], c='red', s=50, marker='D') #cluster 1
                ax[i, j].set_xlim(-1,1)
                ax[i, j].set_ylim(-1,1)
                ax[i, j].grid(True, fillstyle='full')
            
    fig.savefig(fig_dir+'/centroids_compare_'+sample_id+'.png')
    plt.show()
    
def plot_clusters(latent_coords, cluster_assignments, labels=['BG', 'SIG'], cluster_centers=None, title_suffix=None, filename_suffix=None, save_dir=None):

    """
        Only for artificially generated data --> not used for particles
    """
    
    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim
    nrows, ncols = u.calculate_nrows_ncols(latent_dim_n)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)

    for d, ax in zip(range(0, latent_dim_n, 2), axs.flat if latent_dim_n > 2 else [axs]):
        scatter = ax.scatter(latent_coords[:,d], latent_coords[:,d+1], c=cluster_assignments, s=100, marker="o", cmap='Dark2')
        ax.set_title(r'$z_{} \quad & \quad z_{}$'.format(d+1, d+2), fontsize='small')
        if cluster_centers is not None:
            ax.scatter(cluster_centers[:, d], cluster_centers[:, d+1], c='black', s=100, alpha=0.5);

    if latent_dim_n > 2 and axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    ax.add_artist(legend1)

    plt.suptitle(' '.join(filter(None, ['data', title_suffix])))
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, '_'.join(filter(None, ['clustering', filename_suffix, '.png']))))
    else:
        plt.show()
    plt.close(fig)
    
def plot_distance_to_centroids_quantum(data, name_signal='GtWWna35', normalize=False, id_fig=None, save_dir=None):
    dist, dist_s, cluster_label, cluster_label_s = data
    
    if normalize:
        dist = ut.normalize(dist)
        dist_s = ut.normalize(dist_s)
    
    for i in range(0, dist.shape[1]): # second dim = number of clusters
        figure = plt.figure()
        plt.hist(dist[cluster_label==i,i], histtype = 'step', fill=True, bins=100, label='QCD signal', alpha=0.75, density=True, color='Orange')
        
        plt.hist(dist_s[cluster_label_s==i,i], histtype = 'step', fill=False, bins=100, label=f'{name_signal}', density=True, color='deepskyblue')
        #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
        plt.yscale('log')
        plt.legend()
        plt.title(f'Quantum Distance to cluster median {i}')
        if save_dir: figure.savefig(f'{save_dir}/quantum_distance_{id_fig}_cluster{i}.png')
        plt.show()
        
def plot_distance_to_centroids_classic(data, name_signal='GtWWna35', normalize=False, id_fig=None, save_dir=None):
    dist, dist_s, cluster_label, cluster_label_s = data
    
    if normalize:
        dist = ut.normalize(dist)
        dist_s = ut.normalize(dist_s)
    
    for i in range(0, dist.shape[1]): # second dim = number of clusters
        figure = plt.figure()
        plt.hist(dist[cluster_label==i,i], histtype = 'step', fill=True, bins=100, label='QCD signal', alpha=0.75, density=True, color='Orange')
        
        plt.hist(dist_s[cluster_label_s==i,i], histtype = 'step', fill=False, bins=100, label=f'{name_signal}', density=True, color='deepskyblue')
        #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
        plt.yscale('log')
        plt.legend()
        plt.title(f'Euclidian Distance to cluster median {i}')
        if save_dir: figure.savefig(f'{save_dir}/euclidian_distance_{id_fig}_cluster{i}.png')
        plt.show()
        
def plot_distance_to_centroids_compare(data, name_signal='GtWWna35', test=True, normalize=False, id_fig=None, save_dir=None):
    if test: dist_q, dist_qs, cluster_label_q, cluster_label_qs, dist_c, dist_cs, cluster_label_c, cluster_label_cs = data
    else: dist_q, cluster_label_q, dist_c, cluster_label_c = data
    
    if normalize:
        dist_q = ut.normalize(dist_q)
        dist_c = ut.normalize(dist_c)
        if test:
            dist_qs = ut.normalize(dist_qs)
            dist_cs = ut.normalize(dist_cs)
    
    for i in range(0, dist_q.shape[1]): # second dim = number of clusters
        plt.figure()
        plt.hist(dist_q[cluster_label_q==i,i], histtype = 'step', fill=False, bins=100, label='QCD signal (Q)', density=True, color='darkviolet')
        plt.hist(dist_c[cluster_label_c==i,i], histtype = 'step', fill=False, bins=100, label='QCD signal (C)', density=True, color='forestgreen')
        
        if test:
            plt.hist(dist_qs[cluster_label_qs==i,i], histtype = 'step', fill=False, bins=100, label=f'{name_signal} (Q)', alpha=0.75, density=True, color='darkviolet')
            plt.hist(dist_cs[cluster_label_cs==i,i], histtype = 'step', fill=False, bins=100, label=f'{name_signal} (C)', alpha=0.75, density=True, color='forestgreen')
        #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
        plt.yscale('log')
        plt.legend()
        plt.title(f'Quantum vs Euclidian Distance to cluster median {i}')
        if save_dir: figure.savefig(f'{save_dir}/QvsC_distance_{id_fig}_cluster{i}.png')
        plt.show()
        
def plot_sum_distance_compare(data, name_signal='GtWWna35', normalize=False, id_fig=None, save_dir=None):
    dist_q, dist_qs, dist_c, dist_cs = data
    
    if normalize:
        dist_q = ut.normalize(dist_q)
        dist_c = ut.normalize(dist_c)
        dist_qs = ut.normalize(dist_qs)
        dist_cs = ut.normalize(dist_cs)
    # sum distances
    dist_q = np.sum(dist_q, axis=1)
    dist_c = np.sum(dist_c, axis=1)
    dist_qs = np.sum(dist_qs, axis=1)
    dist_cs = np.sum(dist_cs, axis=1)
        
    fig1 = plt.figure(figsize=(8,6))
    plt.hist(dist_c, histtype = 'step', fill=True, linewidth=2, bins=60, label='QCD signal (C)', density=True,alpha=0.55, color='forestgreen',range=(0,8))

    plt.hist(dist_cs, histtype = 'step', fill=False, linewidth=2, bins=60, label=f'{name_signal} (C)', density=True, color='darkviolet', range=(0,8))
    #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
    plt.yscale('log')
    plt.legend(prop={'size': 11}, frameon=True)
    plt.title(f'Euclidian Sum of Distances to cluster medians')
    if save_dir: fig1.savefig(f'{save_dir}/C_sum_distance_{id_fig}.png')
    plt.show()
    
    fig2 = plt.figure(figsize=(8,6))
    plt.hist(dist_q, histtype = 'step', fill=True, linewidth=2, bins=60, label='QCD signal (Q)', alpha=0.55, density=True, color='forestgreen',range=(0,8))

    plt.hist(dist_qs, histtype = 'step', fill=False, linewidth=2, bins=60, label=f'{name_signal} (Q)', density=True, color='darkviolet',range=(0,8))
    #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
    plt.yscale('log')
    plt.legend(prop={'size': 11}, frameon=True)
    plt.title(f'Quantum Sum of Distances to cluster medians')
    if save_dir: fig2.savefig(f'{save_dir}/Q_sum_distance_{id_fig}.png')
    plt.show()
    
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

def plot_rocs_QKmedians(background, signal, title, save_dir=None):
    dist_q, dist_c = background
    dist_qs, dist_cs = signal
    
    # quantum data
    data_q = get_roc_data(np.sum(dist_q,axis=1), np.sum(dist_qs,axis=1))
    # classic data
    data_c = get_roc_data(np.sum(dist_c,axis=1), np.sum(dist_cs,axis=1))
    
    fig = plt.figure(figsize=(8,8))
    plt.loglog(data_q[1], 1.0/data_q[0], label='%s: (auc = %.2f)'% ('quantum k-medians', data_q[2]*100.), linewidth=1.5, color='darkviolet')
    plt.loglog(data_c[1], 1.0/data_c[0], label='%s: (auc = %.2f)'% ('classical k-medians', data_c[2]*100.), linewidth=1.5, color='forestgreen')
    
    #plt.yscale('log', nonpositive='clip')
    #plt.xscale('log', nonpositive='clip')
    plt.ylabel('1/FPR')
    plt.xlabel('TPR')
    plt.title(title)
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True)
    #plt.show()
    plt.savefig(f'{save_dir}/ROC_Kmedians_QvsC_4000B_3200S_DI.pdf', dpi = fig.dpi, bbox_inches='tight')

def plot_rocs_QKmedians_compare(background, signal, n, colors, ids, title, legend_loc='best', ix=None, save_dir=None):
    dist_q=[]; dist_c=[]
    dist_qs=[]; dist_cs=[]
    
    for i in range(n):
        dq, dc = background[i]
        dqs, dcs = signal[i]
        dist_q.append(dq)
        dist_c.append(dc)
        dist_qs.append(dqs)
        dist_cs.append(dcs)
    
    fig = plt.figure(figsize=(8,8))

    for i in range(n):
        # quantum data
        data_q = get_roc_data(np.sum(dist_q[i],axis=1), np.sum(dist_qs[i],axis=1))
        # classic data
        data_c = get_roc_data(np.sum(dist_c[i],axis=1), np.sum(dist_cs[i],axis=1))
        xq = data_q[1]; yq = data_q[0]
        xc = data_c[1]; yc = data_c[0]
        # errors_q = np.sqrt((xq*(1-yq))/len(xq))
        # errors_c = np.sqrt((xc*(1-yc))/len(xc))
        
        plt.plot(xq, 1./yq, label='(%s) Quantum: (auc = %.2f)'% (ids[i], data_q[2]*100.), linewidth=1.5, color=colors[i])
        #plt.errorbar(xq, 1./yq, yerr=1./(errors_q*yq), label='quantum k-med (%s): (auc = %.2f)'% (latent_dims[i], data_q[2]*100.), linewidth=1.5, color=colors[i], ecolor='red', uplims=True, lolims=True)
        plt.plot(xc, 1./yc, label='(%s) Classic: (auc = %.2f)'% (ids[i], data_c[2]*100.), linewidth=1.5, color=colors[i], linestyle='dashed')
        #plt.errorbar(xc, 1./yc, yerr=1./(errors_c*yc), label='classical k-med (%s): (auc = %.2f)'% (latent_dims[i], data_c[2]*100.), linewidth=1.5, color=colors[i], linestyle='dashed', ecolor='black', uplims=True, lolims=True)
    plt.ylabel('1/FPR')
    plt.xlabel('TPR')
    plt.yscale('log')
    #plt.xscale('log')
    #plt.ylim((0, 10**6))
    #plt.xscale('log')
    #plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    #plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
    plt.title(title)
    leg = plt.legend(title=r'$Lat.Dim.$', fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
    leg.get_title().set_position((-40, 0))
    #leg._legend_box.align = "left"
    plt.grid(True)
    #plt.show()

    plt.savefig(f'{save_dir}/ROC_QvsC_Kmedians_1overFPR_TPR_{ix}.pdf', dpi = fig.dpi, bbox_inches='tight')

def parse_tpr_window(string):

    string = string.split(',')
    s1 = string[0].split('.')
    s1 = s1[1]
    s2 = string[1].split('.')
    s2 = s2[1].split(']')
    s2 = s2[0]
    
    return s1, s2
    
def plot_roc_analysis(input_q, input_c, ids, xlabel, ylabel, plot='auc', title=None, legend_loc='best', save_dir=None, ix=None):
    numbers = list(range(len(ids)))
    # quantum data and error
    data_q = [i[0] for i in input_q]
    err_q = [i[1] for i in input_q]
    # classical data and error
    data_c = [i[0] for i in input_c]
    err_c = [i[1] for i in input_c]
    
    fig = plt.figure(figsize=(10,8))
    plt.errorbar(numbers, data_q, yerr=err_q, label='Quantum',
            linestyle='None', marker='o', capsize=3, color='coral')
    plt.errorbar(numbers, data_c, yerr=err_c, label='Classic',
            linestyle='None', marker='v', capsize=3, color='forestgreen')
    # plt.plot(numbers, auc_q, label='quantum', linewidth=1.5, color=colors[0])
    # plt.plot(numbers, auc_c, label='classical', linewidth=1.5, color=colors[1])
    plt.xticks(numbers, ids, fontsize=13)
    if plot=='auc': plt.ylim(0.7, 1.0)
    else: plt.ylim(0.0, 200)
    plt.yticks(fontsize=13)
    plt.xlabel(xlabel, fontsize=15, loc='center')
    plt.ylabel(ylabel, fontsize=15, loc='center')
    #plt.yscale('log')
    if title:
        plt.title(f'TPR window: {title}', fontsize=20)
    leg = plt.legend(loc=f'{legend_loc}', fancybox=True, frameon=True, prop={"size":10})
    plt.grid(True)
    #plt.show()
    if title:
        l1, l2 = parse_tpr_window(title)
        plt.savefig(f'{save_dir}/1overFPR_vs_{xlabel}_{ix}_TPR{l1}{l2}.pdf', dpi = fig.dpi, bbox_inches='tight')
    else: plt.savefig(f'{save_dir}/{ylabel}_vs_{xlabel}_{ix}.pdf', dpi = fig.dpi, bbox_inches='tight')

    
def plot_ROCs_compare(quantum, classic, ids, colors, title, xlabel='TPR', ylabel='1/FPR', legend_loc='best', legend_title='$Minimization$', save_dir=None):
    
    fig = plt.figure(figsize=(8,8))
    for i in range(len(ids)): # for each latent space or train size
        quantum_loss_qcd, quantum_loss_sig = quantum[i]
        classic_loss_qcd, classic_loss_sig = classic[i]
        
        # quantum data
        data_q = get_roc_data(quantum_loss_qcd, quantum_loss_sig)
        # classic data
        data_c = get_roc_data(classic_loss_qcd, classic_loss_sig)
        xq = data_q[1]; yq = data_q[0]
        xc = data_c[1]; yc = data_c[0]
        
        
        plt.plot(xq, 1./yq, label='(%s) Q - old: (auc = %.2f)'% (ids[i], data_q[2]*100.), linewidth=1.5, color=colors[i])
        plt.plot(xc, 1./yc, label='(%s) Q - new: (auc = %.2f)'% (ids[i], data_c[2]*100.), linewidth=1.5, color=colors[i], linestyle='dashed')
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yscale('log')
    #plt.xscale('log')
    #plt.xlim((10**(-2), 0))
    #plt.ylim(0, 10**2)
    #plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    #plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
    plt.title(title)
    leg = plt.legend(fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
    leg.get_title().set_position((-40, 0))
    plt.grid(True)

    if save_dir:
        plt.savefig(f'{save_dir}/ROC_QvsC_Kmedians_latent_compareMedianCalc.pdf', dpi = fig.dpi, bbox_inches='tight')
    else:
        plt.show()
        
def plot_correlation_pT_AD_score(read_dir, ids, n_samples_train, signal_name, mass, br_na=None, save_dir=None):
    import matplotlib as mpl
    label_size = 10
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size

    for x, i in enumerate(ids):
        with h5py.File(f'{read_dir}/{i}/Latent_{i}_trainsize_{n_samples_train[x]}_{signal_name}{mass}{br_na}_pTcorr.h5', 'r') as file:
            q_score_qcd = np.array(file['quantum_loss_qcd'])
            q_score_qcd_pT = np.array(file['quantum_loss_qcd_pT'])
            q_score_sig = np.array(file['quantum_loss_sig'])
            q_score_sig_pT = np.array(file['quantum_loss_sig_pT'])
            c_score_qcd = np.array(file['classic_loss_qcd'])
            c_score_qcd_pT = np.array(file['classic_loss_qcd_pT'])
            c_score_sig = np.array(file['classic_loss_sig'])
            c_score_sig_pT = np.array(file['classic_loss_sig_pT'])
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].hist2d(q_score_qcd, q_score_qcd_pT, density=True, bins=100, range=((0,8),(0, 2500)), norm=mpl.colors.LogNorm())
        axs[0, 0].text(5, 2200, 'corr = %.3f ' % u.pearson_coef(q_score_qcd, q_score_qcd_pT), bbox=dict(facecolor='red', alpha=0.5), fontsize=10)
        axs[0, 0].set_title('pT vs. quantum score QCD', fontsize=12)
        axs[0, 1].hist2d(q_score_sig, q_score_sig_pT, density=True, bins=100, range=((0,8),(0, 2500)), norm=mpl.colors.LogNorm())
        axs[0, 1].text(5, 2200, 'corr = %.3f '  % u.pearson_coef(q_score_sig, q_score_sig_pT), bbox=dict(facecolor='red', alpha=0.5), fontsize=10)
        if br_na: axs[0, 1].set_title(f'pT vs. quantum score {signal_name}_{mass}_{br_na}', fontsize=12)
        else: axs[0, 1].set_title(f'pT vs. quantum score {signal_name}_{mass}', fontsize=12)
        axs[1, 0].hist2d(c_score_qcd, c_score_qcd_pT, density=True, bins=100, range=((0,8),(0, 2500)), norm=mpl.colors.LogNorm())
        axs[1, 0].text(5, 2200, 'corr = %.3f ' % u.pearson_coef(c_score_qcd, c_score_qcd_pT), bbox=dict(facecolor='red', alpha=0.5), fontsize=10)
        axs[1, 0].set_title('pT vs. classic score QCD', fontsize=12)
        axs[1, 1].hist2d(c_score_sig, c_score_sig_pT, density=True, bins=100, range=((0,8),(0, 2500)), norm=mpl.colors.LogNorm())
        axs[1, 1].text(5, 2200, 'corr = %.3f ' % u.pearson_coef(c_score_sig, c_score_sig_pT), bbox=dict(facecolor='red', alpha=0.5), fontsize=10)
        if br_na: axs[1, 1].set_title(f'pT vs. classic score {signal_name}_{mass}_{br_na}', fontsize=12)
        else: axs[1, 1].set_title(f'pT vs. classic score {signal_name}_{mass}', fontsize=12)

        axs[0,0].set(ylabel='pT jet')
        axs[1,1].set(xlabel='AD score')
        axs[0,0].xaxis.set_ticks([])
        axs[0,1].xaxis.set_ticks([]); axs[0,1].yaxis.set_ticks([])
        axs[1,1].yaxis.set_ticks([])

        if save_dir: fig.savefig(f'{save_dir}/correlations_pT_AD_score_lat{i}_trainsize{n_samples_train[x]}.pdf')
        else: plt.show()
        
def plot_auc_fpr(quantum, classic, n, ids, xlabel, tpr_window= [0.5, 0.7], title=None, colors=['C11', 'C12'], legend_loc='best', ix=None, save_dir=None):
    
    # auc_q=[]; auc_c=[]
    one_over_fpr_q=[]; one_over_fpr_c=[]
    for i in range(n):
        quantum_loss_qcd, quantum_loss_sig = quantum[i]
        classic_loss_qcd, classic_loss_sig = classic[i]
        
        metric_q = u.get_metric(quantum_loss_qcd, quantum_loss_sig, tpr_window=tpr_window)
        metric_c = u.get_metric(classic_loss_qcd, classic_loss_sig, tpr_window=tpr_window)
        
        # auc_q.append(metric_q[0])
        # auc_c.append(metric_c[0])
        
        one_over_fpr_q.append(metric_q)
        one_over_fpr_c.append(metric_c)
        
        # # quantum data
        # data_q = get_roc_data(np.sum(dq,axis=1), np.sum(dqs,axis=1))
        # # classic data
        # data_c = get_roc_data(np.sum(dc,axis=1), np.sum(dcs,axis=1))
        # auc_q.append(data_q[2])
        # auc_c.append(data_c[2])
    
    # plot_roc_analysis(auc_q, auc_c, ids=ids, xlabel=xlabel, ylabel='AUC', save_dir=save_dir, ix=ix)
    plot_roc_analysis(one_over_fpr_q, one_over_fpr_c, ids=ids, xlabel=xlabel, ylabel='1/FPR', plot='1/fpr', title=str(tpr_window), save_dir=save_dir, ix=ix)

def divide_error(numerator, denumerator):
    val = numerator[0]/denumerator[0]
    val_error = val * np.sqrt((numerator[1]/numerator[0])**2 + (denumerator[1]/denumerator[0])**2)

    return (val, val_error)    

def plot_ratio_QC_auc_kfold(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, xlabel='Latent dimensions', title=None, legend_loc='best', save_dir=None):
    
    # auc_data_q = []; auc_data_c = []
    # auc_err_q=[]; auc_err_c=[]
    ratio = []; ratio_err=[]
    
    for i in range(len(ids)): # for each latent space or train size
        auc_q=[]; auc_c=[]
        for j in range(n_folds):
            # quantum data
            _,_,aq = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
            # classic data
            _,_,ac = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])

            auc_q.append(aq)
            auc_c.append(ac)
        auc_mean_q = np.mean(auc_q)
        auc_std_q = np.std(auc_q)
        
        auc_mean_c = np.mean(auc_c)
        auc_std_c = np.std(auc_c)
        
        r, r_err = divide_error((auc_mean_q, auc_std_q), (auc_mean_c, auc_std_c)) # find ratio value and error
        
        ratio.append(r)
        ratio_err.append(r_err)
        # auc_data_q.append(auc_mean_q)
        # auc_err_q.append(auc_std_q)
        # auc_data_c.append(auc_mean_c)
        # auc_err_c.append(auc_std_c)
    
    numbers = list(range(len(ids)))
    fig = plt.figure(figsize=(10,8))
    plt.errorbar(numbers, ratio, yerr=ratio_err,
            linestyle='None', marker='o', capsize=3, color='coral')
    plt.xticks(numbers, ids, fontsize=13)
    plt.ylim(0.5, 1.5)
    plt.yticks(fontsize=13)
    plt.xlabel(xlabel, fontsize=15, loc='center')
    plt.ylabel('AUC ratio Q/C', fontsize=15, loc='center')
    #plt.yscale('log')
    #leg = plt.legend(loc=f'{legend_loc}', fancybox=True, frameon=True, prop={"size":10})
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/ratioAUC_vs_{xlabel}_kfold.pdf', dpi = fig.dpi, bbox_inches='tight')
    else: plt.show()

def get_mean_and_error(data):
    return [np.mean(data, axis=0), np.std(data, axis=0)]

def get_FPR(tpr_loss, threshold_loss, tpr_window):
    position = np.where((tpr_loss>=tpr_window[0]) & (tpr_loss<=tpr_window[1]))[0][0]
    threshold_data = threshold_loss[position]
    pred_data = [1 if i>= threshold_data else 0 for i in list(pred_val)]
    tn, fp, fn, tp = confusion_matrix(true_val, pred_data).ravel()
    fpr_data = fp / (fp + tn)
    return fpr_data

def get_auc(fpr_list, tpr_list):
    from scipy import integrate
    sorted_index = np.argsort(fpr_list)
    fpr_list_sorted =  np.array(fpr_list)[sorted_index]
    tpr_list_sorted = np.array(tpr_list)[sorted_index]
    return integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)

def plot_ROC_kfold(quantum_loss_qcd, quantum_loss_sig, classic_loss_qcd, classic_loss_sig, ids, n_folds, colors, title, pic_id, xlabel='TPR', ylabel='FPR', legend_loc='best', legend_title='$ROC$', save_dir=None):

    fig = plt.figure(figsize=(12,10))

    for i in range(len(ids)): # for each latent space or train size
        fpr_q=[]; fpr_c=[]
        auc_q=[]; auc_c=[]
        tpr_q=[]; tpr_c=[]
        one_over_fpr_q=[]; one_over_fpr_c=[]
        for j in range(n_folds):
            # quantum data
            fq, tq, _ = get_roc_data(quantum_loss_qcd[i][j], quantum_loss_sig[i][j])
            # classic data
            fc, tc, _ = get_roc_data(classic_loss_qcd[i][j], classic_loss_sig[i][j])
            auc_q.append(auc(fq, tq)); auc_c.append(auc(fc, tc))
            #one_over_fpr_q.append(1./np.array(fq)); one_over_fpr_c.append(1./np.array(fc))
            fpr_q.append(fq); fpr_c.append(fc)
            tpr_q.append(tq); tpr_c.append(tc)
        
        auc_data_q = get_mean_and_error(np.array(auc_q))
        auc_data_c = get_mean_and_error(np.array(auc_c))
        
        fpr_data_q = get_mean_and_error(np.array(fpr_q))
        fpr_data_c = get_mean_and_error(np.array(fpr_c))
        #print(np.array(tpr_q).shape)
        tpr_mean_q = np.mean(np.array(tpr_q), axis=0)
        print(tpr_mean_q.shape)
        # if i==1: 
        #     print('TPR mean min value: '+ str(min(tpr_mean_q))+ ', index: '+str(np.argmin(tpr_mean_q)))
        #     print('FPR mean max value when cuttting on 0.6: '+ str(max(fpr_data_q[0][tpr_mean_q<0.5])))
        #     print('FPR value for min TPR mean value: '+str(fpr_data_q[0][int(np.argmin(tpr_mean_q))]))
        tpr_mean_c = np.mean(np.array(tpr_c), axis=0)
        #plt.fill_between(x, y-error, y+error)
        
        one_over_fpr_error = fpr_data_q[1]*(1./np.power(fpr_data_q[0],2))
        #one_over_fpr_error = np.std(fpr_data_q[0], axis=0)
        plt.plot(tpr_mean_q, 1./fpr_data_q[0],  linewidth=1.5, color=colors[i], label='(%s) Quantum: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_q[0]*100., auc_data_q[1]*100.))
        plt.fill_between(tpr_mean_q, 1./fpr_data_q[0]-fpr_data_q[1], 1./fpr_data_q[0]+fpr_data_q[1], alpha=0.2, color=colors[i])
        # plt.plot(tpr_mean_c, fpr_data_c[0], '--', linewidth=1.5, color=colors[i], label='(%s) Classic: (auc = %.2f+/-%.2f)'% (ids[i], auc_data_c[0]*100., auc_data_c[1]*100.))
        # plt.fill_between(tpr_mean_c, fpr_data_c[0]-fpr_data_c[1], fpr_data_c[0]+fpr_data_c[1], alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yscale('log')
    #plt.xscale('log')
    #plt.xlim(0.2, 1.0)
    plt.title(title)
    leg = plt.legend(fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
    leg.get_title().set_position((-40, 0))
    #fig.tight_layout()
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/ROC_final_{pic_id}.pdf', dpi = fig.dpi, bbox_inches='tight')
    else: plt.show()

def calculate_ROCs_kfold(runs, n_samples_train, identifiers, n_fold=4, lat_dim=None, qcd_test_size=500, n_samples_signal=500, signal_name='RSGraviton_WW_NA', mass='3.5', br_na=None):
    background_total=[]; signal_total=[]
    
    for i in range(len(runs)):
        background=[]; signal=[]
        for j in range(n_fold):
            centroids = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/centroids/centroids_{runs[i]}_DI_AE_{str(n_samples_train[i])}_correctedcuts_centroids_conv_{j+1}.npy')
            
            save_dir = '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts'
            centroids_c_file = f'{save_dir}/centroids/centroids_{runs[i]}_{str(n_samples_train[i])}_classic_centroids_conv_{j+1}.npy'
            loss_c_file=f'{save_dir}/LOSS_{runs[i]}_{str(n_samples_train[i])}_classic_centroids_conv_{j+1}.npy'
            
            data_qcd, data_s, centroids_c, loss_c = u.load_data_and_centroids_c(runs[i], i=j+1, n_samples_train=n_samples_train[i], qcd_test_size=qcd_test_size, n_samples_test=n_samples_signal, signal_name=signal_name, mass=mass, br_na=br_na, centroids_c_dir=centroids_c_file, loss_c_dir=loss_c_file)
            
            #np.save(centroids_c_file, loss_c)
            #np.save(loss_c_file, centroids_c)
            
            _, q_distances = qkmed.find_nearest_neighbour_DI(data_qcd, centroids)
            _, q_distances_s = qkmed.find_nearest_neighbour_DI(data_s,centroids)
            _, c_distances = cf.find_nearest_neighbour_classic(data_qcd,centroids_c)
            _, c_distances_s = cf.find_nearest_neighbour_classic(data_s,centroids_c)
            
            background.append([q_distances, c_distances])
            signal.append([q_distances_s, c_distances_s])
        background_total.append(background)
        signal_total.append(signal)
    
    return background_total, signal_total

def calculate_ROCs(runs, n_samples_train, identifiers, lat_dim=None, qcd_test_size=500, n_samples_signal=500, br_na=None, signal_name='RSGraviton_WW_NA', mass='3.5', load_filename=None, around_peak=None):
    """
        run_i and lat_dim - identify the latent space dimension
    """
    #cluster_labels=[]; centroids=[]; data=[]
    background=[]; signal=[]
    for i in range(len(runs)):
        #cluster_labels = np.load(f'cluster_label_{runs[i]}_Durr_DI_AE_{n_samples_train[i]}.npy')
        if load_filename:
            print("Centroids loaded")
            centroids = np.load(load_filename)
        elif lat_dim:
            centroids = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_{runs[i]}_Durr_DI_AE_{str(n_samples_train[i])}_lat{lat_dim}.npy')
        elif 'argmin' in identifiers[i]:
            centroids = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_{runs[i]}_Durr_DI_AE_{str(n_samples_train[i])}_minClassic.npy')
        elif 'Grover' in identifiers[i]:
            centroids = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/centroids/centroids_{runs[i]}_DI_AE_{str(n_samples_train[i])}_correctedcuts_centroids_conv_GROVER.npy')
        else:
            print("Centroids loaded default!")
            centroids = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/centroids/centroids_{runs[i]}_argmin_DI_AE_{str(n_samples_train[i])}_correctedcuts_centroids_conv.npy')
        
        data_qcd, data_s, centroids_c, loss_c = u.load_data_and_centroids_c(runs[i], n_samples_train=n_samples_train[i], qcd_test_size=qcd_test_size, n_samples_test=n_samples_signal, signal_name=signal_name, mass=mass, br_na=br_na, around_peak=around_peak)
        
        save_dir = '/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts'
        
        np.save(f'{save_dir}/LOSS_{runs[i]}_{str(n_samples_train[i])}_classic_centroids_conv.npy', loss_c)
        np.save(f'{save_dir}/centroids/centroids_{runs[i]}_{str(n_samples_train[i])}_classic_centroids_conv.npy', centroids_c)
        #plot_centroids_compare(centroids, centroids_c, f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/trainsizestudy_lat4_try2', f'lat{lat_dim}_ntrain{str(n_samples_train[i])}', clusters=2)
        
        q_cluster_assign, q_distances = qkmed.find_nearest_neighbour_DI(data_qcd, centroids)
        q_cluster_assign_s, q_distances_s = qkmed.find_nearest_neighbour_DI(data_s,centroids)
        c_cluster_assign, c_distances = cf.find_nearest_neighbour_classic(data_qcd,centroids_c)
        c_cluster_assign_s, c_distances_s = cf.find_nearest_neighbour_classic(data_s,centroids_c)
        
        background.append([q_distances, c_distances])
        signal.append([q_distances_s, c_distances_s])
    
    #plot_rocs_QKmedians_compare(background, signal, legend_loc='lower left', ix=id_fig, n=len(runs), colors=np.array(['C'+str(j+1)for j in range(len(runs))]), ids=identifiers, title=title, save_dir=save_fig_dir)
    #plot_train_size_impact(background, signal, ids=identifiers, title='AUC vs Train Size', n=len(runs), ix=id_fig, save_dir=save_fig_dir)
    return background, signal

def AD_scores_q_c(test_qcd, test_sig, centroids_q, centroids_c):
        
     # find cluster assignments + distance to centroids for test data
    q_cluster_assign, q_distances = qkmed.find_nearest_neighbour_DI(test_qcd, centroids_q)
    #plot_latent_representations(test_qcd, q_cluster_assign)
    q_cluster_assign_s, q_distances_s = qkmed.find_nearest_neighbour_DI(test_sig,centroids_q)
    #plot_latent_representations(test_sig, q_cluster_assign_s)
    c_cluster_assign, c_distances = cf.find_nearest_neighbour_classic(test_qcd,centroids_c)
    c_cluster_assign_s, c_distances_s = cf.find_nearest_neighbour_classic(test_sig,centroids_c)

    # calc AD scores
    q_score_qcd = u.ad_score(q_cluster_assign, q_distances)
    q_score_sig = u.ad_score(q_cluster_assign_s, q_distances_s)
    c_score_qcd = u.ad_score(c_cluster_assign, c_distances)
    c_score_sig = u.ad_score(c_cluster_assign_s, c_distances_s)

    # calculate loss from 2 jets
    quantum_loss_qcd, index_min_qlqcd = u.combine_loss_min(q_score_qcd)
    quantum_loss_sig, index_min_qlsig = u.combine_loss_min(q_score_sig)
    quantum = [quantum_loss_qcd, quantum_loss_sig]

    classic_loss_qcd, index_min_clqcd = u.combine_loss_min(c_score_qcd)
    classic_loss_sig, index_min_clsig = u.combine_loss_min(c_score_sig)
    classic = [classic_loss_qcd, classic_loss_sig]
    
    return quantum, classic, [index_min_qlqcd, index_min_qlsig], [index_min_clqcd, index_min_clsig]

def calc_AD_scores(identifiers, n_samples_train, k=2, test_size=10000, signal_name='RSGraviton_WW', mass='35', br_na=None, q_dir='results_qmedians/corrected_cuts/diJet', c_dir='results_kmedians/diJet', read_test_dir='/eos/user/e/epuljak/private/epuljak/public/diJet', classic=True, around_peak=None, pTcorr=False, split=False, n_folds=None):
    
    save_dir='/eos/user/e/epuljak/private/epuljak/public/results_paper_Ema'
    quantum=[]; classic=[]
    
    for i in range(len(identifiers)): # for each latent space or train size
        start_time = time.time()
        # load q-centroids
        centroids_q = np.load(f'{q_dir}/centroids/final/centroids_lat{identifiers[i]}_{n_samples_train[i]}_k{k}_new.npy')
        # load c-centroids
        centroids_c = np.load(f'{c_dir}/centroids/final/centroids_lat{identifiers[i]}_{n_samples_train[i]}.npy')
        
        test_qcd, test_sig = u.load_clustering_test_data(identifiers[i], test_size=test_size, k=k, signal_name=signal_name, mass=mass, br_na=br_na, read_dir=read_test_dir, around_peak=around_peak, split=split, n_folds=n_folds)
        #test_qcd, test_sig = u.load_clustering_test_data_iML(identifiers[i], test_size=test_size, k=2, signal_name=signal_name, mass=mass, br_na=br_na)
        
        if split:
            quantum_loss_qcd=[]; quantum_loss_sig=[]
            classic_loss_qcd=[]; classic_loss_sig=[]
            for j in range(n_folds):
                q,c, _, _ = AD_scores_q_c(test_qcd[j], test_sig[j], centroids_q, centroids_c)
                quantum_loss_qcd.append(q[0]); quantum_loss_sig.append(q[1])
                classic_loss_qcd.append(c[0]); classic_loss_sig.append(c[1])
            quantum_loss_qcd = np.array(quantum_loss_qcd)
            quantum_loss_sig = np.array(quantum_loss_sig)
            classic_loss_qcd = np.array(classic_loss_qcd)
            classic_loss_sig = np.array(classic_loss_sig)
            #print(classic_loss_sig.shape)
            quantum.append([quantum_loss_qcd, quantum_loss_sig])
            classic.append([classic_loss_qcd, classic_loss_sig])
            print(f'Save n_folds={n_folds} for id={identifiers[i]} for: time = {(time.time() - start_time)}')
            with h5py.File(f'{save_dir}/{identifiers[i]}/Latent_{identifiers[i]}_trainsize_{n_samples_train[i]}_{signal_name}{mass}{br_na}_nfolds{n_folds}.h5', 'w') as file:
                file.create_dataset('quantum_loss_qcd', data=quantum_loss_qcd)
                file.create_dataset('quantum_loss_sig', data=quantum_loss_sig)
                file.create_dataset('classic_loss_qcd', data=classic_loss_qcd)
                file.create_dataset('classic_loss_sig', data=classic_loss_sig)
            
        else:
            q,c, index_min_q, index_min_c = AD_scores_q_c(test_qcd, test_sig, centroids_q, centroids_c)
            quantum.append(q)
            classic.append(c)
            
        if pTcorr:
            load_pt_dir = f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/pTs'
            # pTs_qcd = np.load(f'{load_pt_dir}/qcd_sig/pT_particles.npy')
            # phis_qcd = np.load(f'{load_pt_dir}/qcd_sig/phi_particles.npy')
            dj_feat_qcd = np.load(f'{load_pt_dir}/qcd_sig/dijet_features_final.npy')
            
            if br_na: 
                # pTs_sig = np.load(f'{load_pt_dir}/{signal_name}/{br_na}/{mass}/pT_particles.npy')
                # phis_sig = np.load(f'{load_pt_dir}/{signal_name}/{br_na}/{mass}/phi_particles.npy')
                dj_feat_sig = np.load(f'{load_pt_dir}/{signal_name}/{br_na}/{mass}/dijet_features_final.npy')
            else: 
                # pTs_sig = np.load(f'{load_pt_dir}/{signal_name}/{mass}/pT_particles.npy')
                # phis_sig = np.load(f'{load_pt_dir}/{signal_name}/{mass}/phi_particles.npy')
                dj_feat_sig = np.load(f'{load_pt_dir}/{signal_name}/{mass}/dijet_features_final.npy')

            
            # pT_q_qcd_particles, phi_q_qcd_particles= u.find_pT_phi_particles_of_min(pTs_qcd, phis_qcd, index_min_qlqcd)
            # pT_q_sig_particles, phi_q_sig_particles= u.find_pT_phi_particles_of_min(pTs_sig, phis_sig, index_min_qlsig)
            # pT_c_qcd_particles, phi_c_qcd_particles = u.find_pT_phi_particles_of_min(pTs_qcd, phis_qcd, index_min_clqcd)
            # pT_c_sig_particles, phi_c_sig_particles = u.find_pT_phi_particles_of_min(pTs_sig, phis_sig, index_min_clsig)
            
            # pT_q_qcd = u.calc_pT_jet(pT_q_qcd_particles, phi_q_qcd_particles)
            # pT_q_sig = u.calc_pT_jet(pT_q_sig_particles, phi_q_sig_particles)
            # pT_c_qcd = u.calc_pT_jet(pT_c_qcd_particles, phi_c_qcd_particles)
            # pT_c_sig = u.calc_pT_jet(pT_c_sig_particles, phi_c_sig_particles)
            print(dj_feat_sig.shape)
            pT_q_qcd = u.find_pT_jet_of_min(dj_feat_qcd, index_min_q[0])
            pT_q_sig = u.find_pT_jet_of_min(dj_feat_sig, index_min_q[1])
            pT_c_qcd = u.find_pT_jet_of_min(dj_feat_qcd, index_min_c[0])
            pT_c_sig = u.find_pT_jet_of_min(dj_feat_sig, index_min_c[1])
            
            with h5py.File(f'{save_dir}/{identifiers[i]}/Latent_{identifiers[i]}_trainsize_{n_samples_train[i]}_{signal_name}{mass}{br_na}_pTcorr.h5', 'w') as file:
                file.create_dataset('quantum_loss_qcd', data=q[0])
                file.create_dataset('quantum_loss_qcd_pT', data=pT_q_qcd)
                file.create_dataset('quantum_loss_sig', data=q[1])
                file.create_dataset('quantum_loss_sig_pT', data=pT_q_sig)
                file.create_dataset('classic_loss_qcd', data=c[0])
                file.create_dataset('classic_loss_qcd_pT', data=pT_c_qcd)
                file.create_dataset('classic_loss_sig', data=c[1])
                file.create_dataset('classic_loss_sig_pT', data=pT_c_sig)

            
            #check pt by plotting
            # u.make_data_dist_plots(pT_q_qcd, pT_q_qcd_2, '$p_T$ (Q qcd)', 100, True, 'Jet')
            # u.make_data_dist_plots(pT_q_sig, pT_q_sig_2, '$p_T$ (Q sig)', 100, True, 'Jet')
            # u.make_data_dist_plots(pT_c_qcd, pT_c_qcd_2, '$p_T$ (C qcd)', 100, True, 'Jet')
            # u.make_data_dist_plots(pT_c_sig, pT_c_sig_2, '$p_T$ (C sig)', 100, True, 'Jet')
        
    return quantum, classic

def calc_AD_scores_nclusters(identifiers, n_samples_train, clusters, k=2, test_size=10000, signal_name='RSGraviton_WW', mass='35', br_na=None, q_dir='results_qmedians/corrected_cuts/diJet', c_dir='results_kmedians/diJet', read_test_dir='/eos/user/e/epuljak/private/epuljak/public/diJet', classic=True, around_peak=None):
    
    save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/ad_scores/final'
    quantum=[]; classic=[]
    
    for i in range(len(clusters)): # for each latent space or train size
        if clusters[i]=='2': 
            centroids_q = np.load(f'{q_dir}/centroids/final/centroids_lat{identifiers[i]}_{n_samples_train[i]}.npy')
            centroids_c = np.load(f'{c_dir}/centroids/final/centroids_lat{identifiers[i]}_{n_samples_train[i]}.npy')
        else: 
            centroids_q = np.load(f'{q_dir}/centroids/final/centroids_lat{identifiers[i]}_{n_samples_train[i]}_k{clusters[i]}.npy')
            centroids_c = np.load(f'{c_dir}/centroids/final/centroids_lat{identifiers[i]}_{n_samples_train[i]}_k{clusters[i]}.npy')
        test_qcd, test_sig = u.load_clustering_test_data(identifiers[i], test_size=test_size, k=k, signal_name=signal_name, mass=mass, br_na=br_na, read_dir=read_test_dir, around_peak=around_peak)
        
        q,c, index_min_q, index_min_c = AD_scores_q_c(test_qcd, test_sig, centroids_q, centroids_c)
        quantum.append(q)
        classic.append(c)
        
    return quantum, classic

def calculate_ROCs_randomVStrained(runs, n_samples_train, identifiers, lat_dim=None, qcd_test_size=500, n_samples_signal=500, br_na=None, signal_name='RSGraviton_WW_NA', mass='3.5'):
    """
        run_i and lat_dim - identify the latent space dimension
    """
    #cluster_labels=[]; centroids=[]; data=[]
    background_Q=[]; signal_Q=[]
    background_C=[]; signal_C=[]
    
    for i in range(len(runs)):
        #---- TRAINED CENTROIDS ----
        if lat_dim:
            centroids_trained_q = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_{runs[i]}_Durr_DI_AE_{str(n_samples_train[i])}_lat{lat_dim}.npy')
        elif 'argmin' in identifiers[i]:
            centroids_trained_q = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_{runs[i]}_Durr_DI_AE_{str(n_samples_train[i])}_minClassic.npy')
        else:
            print("Centroids loaded default!")
            centroids_trained_q = np.load(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/centroids/centroids_{runs[i]}_argmin_DI_AE_{str(n_samples_train[i])}_correctedcuts_centroids_conv.npy')
        
        data_qcd, data_s, centroids_trained_c, _ = u.load_data_and_centroids_c(runs[i], n_samples_train=n_samples_train[i], qcd_test_size=qcd_test_size, n_samples_test=n_samples_signal, signal_name=signal_name, mass=mass, br_na=br_na)
        
        # ---- RANDOM CENTROIDS ----
        # read QCD predicted data (test - SIDE)
        read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/corrected_cuts/{runs[i]}/'
        file_name = 'latentrep_QCD_sig.h5'
        with h5py.File(read_dir+file_name, 'r') as file:
            data = np.array(file['latent_space'][:])
            data_train = data[:n_samples_train[i]]
        
        centroids_random = qkmed.initialize_centroids(data_train, k=2)   # Intialize centroids
        
        # trained Q
        _, q_distances_T = qkmed.find_nearest_neighbour_DI(data_qcd, centroids_trained_q)
        _, q_distances_s_T = qkmed.find_nearest_neighbour_DI(data_s,centroids_trained_q)
        # random q
        _, q_distances_R = qkmed.find_nearest_neighbour_DI(data_qcd, centroids_random)
        _, q_distances_s_R = qkmed.find_nearest_neighbour_DI(data_s,centroids_random)
        
        #trained C
        _, c_distances_T = cf.find_nearest_neighbour_classic(data_qcd,centroids_trained_c)
        _, c_distances_s_T = cf.find_nearest_neighbour_classic(data_s,centroids_trained_c)
        # random C
        _, c_distances_R = cf.find_nearest_neighbour_classic(data_qcd,centroids_random)
        _, c_distances_s_R = cf.find_nearest_neighbour_classic(data_s,centroids_random)
        
        background_Q.append([q_distances_T, q_distances_R])
        signal_Q.append([q_distances_s_T, q_distances_s_R])
        
        background_C.append([c_distances_T, c_distances_R])
        signal_C.append([c_distances_s_T, c_distances_s_R])
    
    #plot_rocs_QKmedians_compare(background, signal, legend_loc='lower left', ix=id_fig, n=len(runs), colors=np.array(['C'+str(j+1)for j in range(len(runs))]), ids=identifiers, title=title, save_dir=save_fig_dir)
    #plot_train_size_impact(background, signal, ids=identifiers, title='AUC vs Train Size', n=len(runs), ix=id_fig, save_dir=save_fig_dir)
    return background_Q, signal_Q, background_C, signal_C