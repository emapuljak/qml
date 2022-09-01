import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
import math
from qibo.models import Circuit
from qibo import gates
#from scripts.util import normalize
from scripts.grover import grover_qc
from scripts.oracle import create_oracle_circ

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, IBMQ, execute, transpile

def duerr_hoyer_algo(distances):
    """
        distance: [1, k] where k is number of cluster centers
    """
    import time
    start_time = time.time()
    
    k = len(distances)
    n = int(math.floor(math.log2(k)) + 1)
    # choose random threshold
    index_rand = np.random.choice(list(range(k)))
    threshold = distances[index_rand]
    max_iters = int(math.ceil(np.sqrt(2**n)))
    for _ in range(max_iters):

        qc = Circuit(n)
    
        for i in range(n):
            qc.add(gates.H(i))
        
        #create oracle
        qc_oracle, n_indices_marked = create_oracle_circ(distances, threshold, n)
        
        #grover circuit
        qc = grover_qc(qc, n, qc_oracle, n_indices_marked)
        with tf.device("/GPU:0"):
            counts = qc.execute(nshots=1000).frequencies(binary=True)
        #measure highest probability
        probs = counts.items()
        sorted_probs = dict(sorted(probs, key=lambda item: item[1], reverse=True))
        sorted_probs_keys = list(sorted_probs.keys())
        new_ix = [int(sorted_probs_keys[i],2) for i, _ in enumerate(sorted_probs_keys) if int(sorted_probs_keys[i],2) < k]
        new_ix = new_ix[0]
        threshold = distances[new_ix]
    #print("Find Min Distance ---> %s seconds ---" % (time.time() - start_time))
    return new_ix

def neg_rotations(point, centroids):
    
    #normalize and encode to [0, pi/2] range
    point = normalize(point)
    point_norm = [math.remainder(a, math.pi/2.) for a in point]
    
    centroids_norm = []
    for centroid in centroids:
        c = normalize(centroid)
        centroids_norm.append([math.remainder(a, math.pi/2.) for a in c])

    n_qubits = centroids.shape[0]
    
    qc = QuantumCircuit(n_qubits, n_qubits, name='neg_rot_circ')
    # rotation for point
    for i in range(n_qubits):
        qc.ry(point_norm, i)
    for i in range(n_qubits):
        qc.ry(centroids_norm[i], i)
    for i in range(n_qubits):
        qc.measure(i, i)
    
    simulator = Aer.get_backend('qasm_simulator')
    circ = transpile(qc, simulator)

    # Run and get counts
    result = simulator.run(qc, shots=1024).result()
    counts = result.get_counts(qc)
    
    #cluster_assignment = counts[]
    return qc, counts
    
    
    
    