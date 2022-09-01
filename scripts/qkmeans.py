import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

from qibo.models import Circuit
from qibo import gates
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, execute, Aer
from qiskit.tools.jupyter import *
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_textbook.tools import vector2latex

import sys
sys.path.append('../')
sys.path.append('../../')
import utils as u
import scripts.minimization as m

def normalize(data):
    return data / np.linalg.norm(data)

def draw_plot(points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=40)
    plt.show()
    
def draw_plot_w_centroids(points, centroids, class_label):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=40, c=class_label)
    plt.scatter(centroids[0,0], centroids[0,1], s=40, c='g', marker='o')
    plt.scatter(centroids[1,0], centroids[1,1], s=40, c='r', marker='o')
    plt.show()

    
def initialize_centroids(points, k):
    return points[np.random.randint(points.shape[0],size=k),:]

def print_state_vector(circ, name):
    state_vec = Statevector.from_instruction(circ).data
    #print(f'State Vector of {name}: {state_vec}')
    #print(f'State Vector of {name}: {Statevector.from_instruction(circ).to_dict()}')
    #vector2latex(state_vec, pretext="|%s\\rangle ="%name)
    #show_figure(plot_bloch_multivector(state_vec))
    return state_vec

def check_norm(state):
    norm = np.power(state,2)/np.linalg.norm(state)**2
    return np.allclose(np.sum(norm), 1.0, rtol=1.e-10)
    
def prepare_input(X: np.ndarray, num_qubits: int, num_features: int):
    """Prepare the input data for clustering.
    Args:
        X: numpy.ndarray of shape (n_features,)
            The input data to be clustered.
    Returns:
        X: numpy.ndarray of shape (2**n_qubits,)
            The input data to be clustered.
    """

    # pad the input data with zeros if the number of features is not a power of 2
    if not float(np.log2(num_features)).is_integer():
        X = np.pad(X, (0, 2 ** num_qubits - num_features), "constant")

    # if all 0's, then return as is
    if np.all(X == 0):
        return X
    # normalize each vector to have unit norm
    X = normalize(X)

    return X

def get_amplitudes_from_qiskit(a, b, num_features):
    """
    Helper function from Qiskit to get amplitudes for initialization of QKmeans circuit
    
    Args:
        a: numpy.ndarray of shape (1, num_features) - point in space
        b: numpy.ndarray of shape (1, num_features) - centroids
        num_features: int
    """
    
    Z = np.linalg.norm(a)**2 + np.linalg.norm(b)**2 # both inputs are normalized
    
    # psi circuit
    ampl_psi = np.concatenate((a/np.linalg.norm(a), b/np.linalg.norm(b)))*(1/np.sqrt(2)) # inputs are normalized
    n_qubits_psi = int(math.ceil(np.log2(len(ampl_psi))))
    qc_psi = QuantumCircuit(n_qubits_psi)
    qc_psi.initialize(ampl_psi, list(range(n_qubits_psi)))
    #state_psi = print_state_vector(qc_psi, 'psi')
    # phi circuit
    ampl_phi = np.array([np.linalg.norm(a), -np.linalg.norm(b)])/np.sqrt(Z)
    n_qubits_phi = 1
    qc_phi = QuantumCircuit(n_qubits_phi) # always 1
    qc_phi.initialize(ampl_phi, [0])
    #state_phi = print_state_vector(qc_phi, 'phi')
    
    anc = QuantumRegister(1, "ancilla")
    qr_psi = QuantumRegister(n_qubits_psi, "psi")
    qr_phi = QuantumRegister(1, "phi") # size always 1
    cr = ClassicalRegister(1, "cr")

    # Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = QuantumCircuit(anc, qr_psi, qr_phi, cr, name="k_means")

    qc.append(qc_psi, qr_psi)
    qc.append(qc_phi, qr_phi)
    state_vec = print_state_vector(qc.reverse_bits(), 'ancillaPsiPhi')

    return state_vec

def DistCalc(a, b, num_features, shots_n=1000):
    """
    Args:
        a: numpy.ndarray of shape (1, num_features) - point in space
        b: numpy.ndarray of shape (1, num_features) - centroids
        num_features: int
        shots_n: int - number of shots to execuite QKmeans Quantum circuit
    Returns:
        quantum distance between a and b
        qc: Circuit from qibo.models.Circuit - QKmeans circuit
    """
    import time
    start_time = time.time()
    
    a_norm = prepare_input(a)
    b_norm = prepare_input(b)

    amplitudes = get_amplitudes_from_qiskit(a_norm, b_norm, num_features)
    ab_ampl_len = len(np.concatenate((a,b)))
    
    # Encode classical data as quantum state - amplitude encoding
    #Z = np.linalg.norm(a)**2 + np.linalg.norm(b)**2
    Z = np.linalg.norm(a_norm)**2 + np.linalg.norm(b_norm)**2 # because both inputs are normalized
    
    # psi circuit
    #ampl_psi = np.concatenate((a, b))*(1/np.sqrt(2))
    n_qubits_psi = int(np.log2(ab_ampl_len))
    qc_psi = Circuit(n_qubits_psi)
    
    # phi circuit
    #ampl_phi = np.array([np.linalg.norm(a), -np.linalg.norm(b)])/np.sqrt(Z)
    n_qubits_phi = 1 # always
    qc_phi = Circuit(n_qubits_phi)
    #qc_phi.add(gates.H(0))

    # create QKmeans circuit
    qc = Circuit(n_qubits_psi+n_qubits_phi+1) # 1 = ancilla, 1 = phi
    
    qc.add(qc_psi.on_qubits(*(list(range(1,n_qubits_psi+n_qubits_phi)))))
    qc.add(qc_phi.on_qubits(n_qubits_psi+n_qubits_phi))
    
    
    # SwapTest
    qc.add(gates.H(0))
    qc.add(gates.SWAP(n_qubits_psi, n_qubits_psi+n_qubits_phi).controlled_by(0)) # swap ancilla of psi and phi
    qc.add(gates.H(0))
    
    qc.add(gates.M(0)) # returing back one number!
    
    result = qc.execute(initial_state=amplitudes, nshots=shots_n)
    counts = result.frequencies(binary=True)
    #show_figure(plot_histogram(counts))
    
    # return overlap, searching for prob. of state 0
    overlap = 2*np.abs(counts['0']/shots_n - 0.5)
    #print("DistCalc ---> %s seconds ---" % (time.time() - start_time))
    return overlap*2*Z, qc


def find_nearest_neighbour(points, centroids):
    import time
    start_time = time.time()
    """
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space;
        centroids: numpy.ndarray of shape (N, X)
    Returns:
        cluster_assignments: numpy.ndarray of shape (N, X) specifying to which cluster each feature is assigned
        distances: numpy.ndarray of shape (N, X) specifying distances to nearest cluster
    """
    
    n = points.shape[0]
    num_features = points.shape[1]
    k = centroids.shape[0] # number of centroids
    #cluster_label = np.zeros(n) # assignment to new centroids
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist, _ = DistCalc(points[i,:], centroids[j,:], num_features, shots_n=10000) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        cluster_index = m.duerr_hoyer_algo(dist)
        print(dist)
        print(cluster_index)
        #cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)

def initialize_centroids(points, k):
    """
    Randomly initialize centroids of data points.
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
        k: int - number of clusters
    """
    return points[np.random.randint(points.shape[0],size=k),:]

def find_centroids(points, cluster_labels, clusters=2):
    """
    Find new cluster centroids by calculating the mean of data points assigned to specific cluster.
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
        cluster_labels: numpy.ndarray of shape (N,)
    Returns:
        new cluster centroids for points
    """
    centroids = np.zeros([clusters,points.shape[1]])
    
    for i in range(clusters):
        centroids[i,:] = points[cluster_labels==i].mean(axis=0)
    
    return np.array(centroids)

