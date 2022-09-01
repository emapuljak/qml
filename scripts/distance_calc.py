import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import tensorflow as tf
from qibo.models import Circuit
from qibo import gates
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, execute, Aer
from qiskit.tools.jupyter import *
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, IBMQ, execute
from qiskit_textbook.tools import vector2latex

from scripts.util import create_gate, prepare_input, normalize, calc_norm

def get_amplitudes_from_qiskit(a_norm, b_norm, n_qubits):
    """
    Helper function from Qiskit to get amplitudes for initialization of QKmedians circuit
    
    Args:
        a: numpy.ndarray of shape (1, num_features) - point in space
        b: numpy.ndarray of shape (1, num_features) - centroids
        num_features: int
    """
    a_con = create_gate(a_norm, n_qubits)
    b_con = create_gate(b_norm, n_qubits)
    
    # defining psi
    psi_con = QuantumRegister(1, name="psi_con")
    psi_state = QuantumRegister(n_qubits, name="psi_state")
    psi = QuantumCircuit(psi_con, psi_state, name="psi")
    psi.h(psi_con)
    psi.append(a_con, psi.qubits)
    psi.x(psi_con)
    psi.append(b_con, psi.qubits)
    
    # defining phi
    phi = QuantumCircuit(n_qubits + 1, name="phi")
    phi.x(0)
    phi.h(0)
    
    Z = 2.0 # both inputs are normalized
    
    anc = QuantumRegister(1, "ancilla")
    qr_psi = QuantumRegister(n_qubits+1, "psi")
    qr_phi = QuantumRegister(n_qubits+1, "phi") # size always 1
    #cr = ClassicalRegister(1, "cr")

    # Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = QuantumCircuit(anc, qr_psi, qr_phi, name="k_means")

    qc.append(psi, qr_psi)
    qc.append(phi, qr_phi)
    state_vec = Statevector.from_instruction(qc.reverse_bits()).data
    return np.asarray(state_vec)

def calc_Z(a, b):
    z = np.linalg.norm(a)**2 + np.linalg.norm(b)**2
    return z

def create_psi(a_norm, b_norm):
    # arrays are normalized
    
    amplitude = np.concatenate((a_norm, b_norm))*(1/np.sqrt(2))
    n_qubits = int(math.ceil(np.log2(len(amplitude))))
    
    psi_state = QuantumRegister(n_qubits, name="psi_state")
    psi = QuantumCircuit(psi_state, name="psi")
    
    psi.initialize(amplitude, psi_state)
    
    return psi

def create_phi(a_norm, b_norm):
    
    amplitude = np.array([1.0, -1.0])/np.sqrt(calc_Z(a_norm, b_norm))
    phi = QuantumCircuit(1, name="phi")
    phi.initialize(amplitude, [0])
    
    return phi
    
    
def get_amplitudes_from_qiskit_AmplE(a_norm, b_norm):
    """
    Helper function from Qiskit to get amplitudes for initialization of QKmedians circuit - Amplitude Embedding
    
    Args:
        a: numpy.ndarray of shape (1, num_features) - point in space
        b: numpy.ndarray of shape (1, num_features) - centroids
        num_features: int
    """
    Z = calc_Z(a_norm, b_norm)
    
    psi = create_psi(a_norm, b_norm)
    phi = create_phi(a_norm, b_norm)
        
    anc = QuantumRegister(1, "ancilla")
    qr_psi = QuantumRegister(psi.width(), "psi")
    qr_phi = QuantumRegister(1, "phi") # size always 1
    #cr = ClassicalRegister(1, "cr")

    # Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = QuantumCircuit(anc, qr_psi, qr_phi, name="k_means")

    qc.append(psi, qr_psi)
    qc.append(phi, qr_phi)
    state_vec = Statevector.from_instruction(qc.reverse_bits()).data
    return np.asarray(state_vec), psi.width()

def DistCalc(a, b, shots_n=10000):
    """
    Args:
        a: numpy.ndarray of shape (1, num_features) - point in space
        b: numpy.ndarray of shape (1, num_features) - centroids
        num_features: int
        shots_n: int - number of shots to execuite QKmedians Quantum circuit
    Returns:
        quantum distance between a and b
        qc: Circuit from qibo.models.Circuit
    """
    import time
    start_time = time.time()
    # calc number of qubits needed
    if float(np.log2(a.shape[0])).is_integer():
        n_qubits = int(np.log2(a.shape[0]))
    else: n_qubits = int(np.log2(a.shape[0])) + 1
    
    a_norm = prepare_input(a, n_qubits, a.shape[0])
    b_norm = prepare_input(b, n_qubits, b.shape[0])
    
    #print(f'Euclidian distance for centroid: {np.linalg.norm(a_norm-b_norm)}')
    amplitudes = get_amplitudes_from_qiskit(a_norm, b_norm, n_qubits)
    list_psi_qubits = list(range(1,n_qubits+1+1))
    list_phi_qubits = list(range(n_qubits+1+1, n_qubits+1+n_qubits+1+1))
    #psi circuit
    qc_psi = Circuit(n_qubits+1)

    # phi circuit
    qc_phi = Circuit(n_qubits+1)

    # create QKmeans circuit
    qc = Circuit(2*(n_qubits+1)+1) # 1 = ancilla psi, 1 = ancilla

    qc.add(qc_psi.on_qubits(*(list_psi_qubits)))
    qc.add(qc_phi.on_qubits(*(list_phi_qubits)))
    
    qc.add(gates.H(0))
    for i, q in enumerate(list_psi_qubits):
        qc.add(gates.SWAP(
            q,
            list_phi_qubits[i]
        ).controlled_by(0))
    qc.add(gates.H(0))
    qc.add(gates.M(0)) # returing back one number!
    with tf.device("/GPU:0"):
        result = qc.execute(initial_state=amplitudes, nshots=shots_n)
    counts = result.frequencies(binary=True)
    #print(f'Result of distance circuit: {counts}')
    Z=2.0
    # return overlap, searching for prob. of state 0
    overlap = 2*np.abs(counts['0']/shots_n - 0.5)
    #print(np.abs(counts['0']/shots_n - 0.5))
    #print(f'Overlap: {overlap}')
    #print("DistCalc ---> %s seconds ---" % (time.time() - start_time))
    #print(f'Quantum distance for centroid: {overlap*2*Z}')
    return overlap*2*Z, qc

def DistCalc_AmplE(a, b, device_name, shots_n=10000):
    """
    Args:
        a: numpy.ndarray of shape (1, num_features) - point in space
        b: numpy.ndarray of shape (1, num_features) - centroids
        num_features: int
        shots_n: int - number of shots to execuite QKmedians Quantum circuit
    Returns:
        quantum distance between a and b
        qc: Circuit from qibo.models.Circuit
    """
    import time
    start_time = time.time()
    
    a_norm = normalize(a)
    b_norm = normalize(b)
    
    #print(f'Euclidian distance for centroid: {np.linalg.norm(a_norm-b_norm)}')
    amplitudes, n_qubits_psi = get_amplitudes_from_qiskit_AmplE(a_norm, b_norm)
    
    #psi circuit
    qc_psi = Circuit(n_qubits_psi)

    # phi circuit
    qc_phi = Circuit(1)

    # create QKmeans circuit
    qc = Circuit(n_qubits_psi+1+1) # 1 = phi, 1 = ancilla
    
    qc.add(qc_psi.on_qubits(*(range(1, n_qubits_psi+1))))
    qc.add(qc_phi.on_qubits(n_qubits_psi+1))
    
    qc.add(gates.H(0))
    qc.add(gates.SWAP(
            n_qubits_psi,
            n_qubits_psi+1
        ).controlled_by(0))
    qc.add(gates.H(0))
    qc.add(gates.M(0)) # returing back one number!
    with tf.device(device_name):
        result = qc.execute(initial_state=amplitudes, nshots=shots_n)
    
    counts = result.frequencies(binary=True)
    #print(f'Result of distance circuit: {counts}')
    Z=calc_Z(a_norm, b_norm)
    # return overlap, searching for prob. of state 0
    overlap = 2*np.abs(counts['0']/shots_n - 0.5)
    #print(np.abs(counts['0']/shots_n - 0.5))
    #print(f'Overlap: {overlap}')
    #print("DistCalc ---> %s seconds ---" % (time.time() - start_time))
    #print(f'Quantum distance for centroid: {overlap*2*Z}')
    return overlap*2*Z, qc

def pad_input(X):
    num_features = len(X)
    if not float(np.log2(num_features)).is_integer():
        size_needed = pow(2, math.ceil(math.log(num_features)/math.log(2)))
        X = np.pad(X, (0, size_needed-num_features), "constant")
    return X

def DistCalc_DI(a, b, device_name='/GPU:0', shots_n=10000):
    """ Distance calculation with destructive interference """
    num_features = len(a)
    norm = calc_norm(a, b)
    a_norm = a/norm
    b_norm = b/norm
    
    a_norm = pad_input(a_norm)
    b_norm = pad_input(b_norm)
    
    amplitudes = np.concatenate((a_norm, b_norm))
    #print(np.array(amplitudes).shape)
    n_qubits = int(np.log2(len(amplitudes)))
    
    #QIBO
    qc = Circuit(n_qubits)
    qc.add(gates.H(0))
    qc.add(gates.M(0))
    with tf.device(device_name):
        result = qc.execute(initial_state=amplitudes, nshots=shots_n)
    counts = result.frequencies(binary=True)
    distance = norm*math.sqrt(2)*math.sqrt((counts['1']/shots_n))
    return distance, qc