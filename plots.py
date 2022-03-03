import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt
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
                ax[i, j].scatter(centroids[0,i], centroids[0,j], c='red', s=50, marker="X") #cluster 0
                ax[i, j].scatter(centroids[1,i], centroids[1,j], c='blue', s=50, marker="D") #cluster 1
                ax[i, j].grid(True, fillstyle='full')
            
    fig.savefig(save_dir+'/centroids_'+sample_id+'.png')
    plt.show()
    
def plot_centroids_compare(centroids_q, centroids_c, save_dir, sample_id, clusters=2):
    
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
                
                ax[i, j].scatter(centroids_c[0,j], centroids_c[0,i], c='indianred', s=50, marker='D') #cluster 0
                ax[i, j].scatter(centroids_c[1,j], centroids_c[1,i], c='maroon', s=50, marker='D') #cluster 1
                ax[i, j].set_xlim(-1,1)
                ax[i, j].set_ylim(-1,1)
                ax[i, j].grid(True, fillstyle='full')
            
    fig.savefig(save_dir+'/centroids_compare_'+sample_id+'.png')
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
    plt.hist(dist_c, histtype = 'step', fill=True, linewidth=2, bins=80, label='QCD signal (C)', density=True,alpha=0.55, color='forestgreen')

    plt.hist(dist_cs, histtype = 'step', fill=False, linewidth=2, bins=80, label=f'{name_signal} (C)', density=True, color='darkviolet')
    #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
    plt.yscale('log')
    plt.legend()
    plt.title(f'Euclidian Sum of Distances to cluster medians')
    if save_dir: fig1.savefig(f'{save_dir}/C_sum_distance_{id_fig}.png')
    plt.show()
    
    fig2 = plt.figure(figsize=(8,6))
    plt.hist(dist_q, histtype = 'step', fill=True, linewidth=2, bins=80, label='QCD signal (Q)', alpha=0.55, density=True, color='forestgreen')

    plt.hist(dist_qs, histtype = 'step', fill=False, linewidth=2, bins=80, label=f'{name_signal} (Q)', density=True, color='darkviolet')
    #plt.hist(ratio[:,i], bins=100, label='ratio c/q', alpha=0.6, density=True)
    plt.yscale('log')
    plt.legend()
    plt.title(f'Quantum Sum of Distances to cluster medians')
    if save_dir: fig2.savefig(f'{save_dir}/Q_sum_distance_{id_fig}.png')
    plt.show()
    
def get_roc_data(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))
    
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data


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
    plt.savefig(f'{save_dir}/ROC_Kmedians_QvsC_500B_400S_DI.pdf', dpi = fig.dpi, bbox_inches='tight')

def plot_rocs_QKmedians_compare(background, signal, n, colors, latent_dims, title, save_dir=None):
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
        
        plt.loglog(data_q[1], 1.0/data_q[0], label='quantum k-med (%s): (auc = %.2f)'% (latent_dims[i], data_q[2]*100.), linewidth=1.5, color=colors[i])
        plt.loglog(data_c[1], 1.0/data_c[0], label='classical k-med (%s): (auc = %.2f)'% (latent_dims[i], data_c[2]*100.), linewidth=1.5, color=colors[i], linestyle='dashed')
    
    plt.ylabel('1/FPR')
    plt.xlabel('TPR')
    #plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    #plt.axvline(0.00001, color='red', linestyle='dashed', linewidth=1)
    plt.title(title)
    plt.legend(loc='lower left', frameon=True, prop={"size":10})
    plt.grid(True)
    #plt.show()

    plt.savefig(f'{save_dir}/ROC_Kmedians_QC_compare_500B_400S_ALL.pdf', dpi = fig.dpi, bbox_inches='tight')

