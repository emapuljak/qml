import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import Initialize
from qiskit.circuit import ControlledGate
import sys
sys.path.append('../')
sys.path.append('../../')
import utils as u
import scripts.minimization as m
import math
import matplotlib.pyplot as plt

def calc_norm(a, b):
    return math.sqrt(np.sum(a**2) + np.sum(b**2))

def normalize(data):
    return np.array(data / np.linalg.norm(data), dtype=np.float32)

def euclidean_dist(a, b):
    return np.linalg.norm(a-b)


def show_figure(fig):
    # See https://github.com/Qiskit/qiskit-terra/issues/1682
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)

def create_gate(x: np.ndarray, num_qubits: int) -> ControlledGate:
    # from qiskit implementation github 
    """Internal method that initializes the state |x>.
    Args:
        x: numpy.ndarray of shape (2**n_qubits,)
            The input data to be initialized.
    Returns:
        `ControlledGate`: The controlled version of the gate that initializes the state |x>.
    """
    # initialize the state
    init_state = Initialize(x)

    # call to generate the circuit that takes the desired vector to zero
    dgc = init_state.gates_to_uncompute()
    # invert the circuit to create the desired vector from zero (assuming
    # the qubits are in the zero state)
    initialize_instr = dgc.to_instruction().inverse()
    q = QuantumRegister(num_qubits, "q")
    initialize_circuit = QuantumCircuit(q, name="x_init")
    initialize_circuit.append(initialize_instr, q[:])
    x_con = initialize_circuit.to_gate().control(num_ctrl_qubits=1, label="x_con")

    return x_con

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
