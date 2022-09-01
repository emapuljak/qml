from qibo.models import Circuit
from qibo import gates
from qibo.gates import Unitary
import numpy as np
import utils as u
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.extensions import UnitaryGate
import matplotlib
import scripts.util as ut

def is_unitary(m):
    return np.allclose(np.identity(m.shape[0]), m.conj().T * m)

def simple_oracle(n, i0):
    """
        n = number of qubits,
        i0 = searched element
    """
    
    oracle_qc = Circuit(n)
    
    # all 1s except from the searched element index
    oracle_matrix = np.matrix(np.identity(2**n))
    # add phase shift to winner index
    i0_idx = int(i0, 2)
    oracle_matrix[i0_idx, i0_idx] = -1
    
    if is_unitary(oracle_matrix):
        oracle_qc.add(Unitary(oracle_matrix, *range(n)))
    else:
        raise Exception('Matrix is not unitary')
    return oracle_qc


def f(x, threshold):
    return x < threshold

def create_oracle_circ(distances, threshold, n_qubits):
    """
    Create circuit for oracle - for one solution --> I - 2*|i0><i0| - extend to multi
        Args:
            distances: numpy.ndarray of shape (num_samples,)
                - Distances to be minimized
            threshold: int
            n_qubits: int
                - number of qubits oracle circuit is applied on.
        Returns:
            qc_oracle: Circuit()
                - oracle circuit
            marked_indices: int
                - number of indices which are marked as lower than threshold
    """
    solutions = []; marked_indices=0
    for index, d in enumerate(distances):
        if f(d, threshold):
            #index_bin = np.binary_repr(index, width=n)
            ket_i0 = np.zeros((2**n_qubits, 1)); ket_i0[index] = 1
            bra_i0 = np.conj(ket_i0).T
            solutions.append(np.dot(ket_i0, bra_i0))
            marked_indices+=1
    if marked_indices==0:
        ket_i0 = np.zeros((2**n_qubits, 1)); ket_i0[np.argmin(distances)] = 1
        bra_i0 = np.conj(ket_i0).T
        solutions.append(np.dot(ket_i0, bra_i0))
        marked_indices+=1
    
    i0 = sum(solutions)
    oracle_matrix = np.identity(2**n_qubits) - 2*i0
    qc_oracle = Circuit(n_qubits)
    qc_oracle.add(Unitary(oracle_matrix, *range(n_qubits)))
    return qc_oracle, marked_indices
    

def create_oracle_circ_QISKIT(distances, threshold, n_qubits):
    """
    Create circuit for oracle - for one solution --> I - 2*|i0><i0|
        Args:
            distances: numpy.ndarray of shape (num_samples,)
                - Distances to be minimized
            threshold: int
            n_qubits: int
                - number of qubits oracle circuit is applied on.
        Returns:
            qc_oracle: Circuit()
                - oracle circuit
            marked_indices: int
                - number of indices which are marked as lower than threshold
    """
    solutions = []; marked_indices=0
    for index, d in enumerate(distances):
        if f(d, threshold):
            #index_bin = np.binary_repr(index, width=n)
            ket_i0 = np.zeros((2**n_qubits, 1)); ket_i0[index] = 1
            bra_i0 = np.conj(ket_i0).T
            solutions.append(np.dot(ket_i0, bra_i0))
            marked_indices+=1
    if marked_indices==0:
        ket_i0 = np.zeros((2**n_qubits, 1)); ket_i0[np.argmin(distances)] = 1
        bra_i0 = np.conj(ket_i0).T
        solutions.append(np.dot(ket_i0, bra_i0))
        marked_indices+=1
    i0 = sum(solutions)
    oracle_matrix = np.identity(2**n_qubits) - 2*i0
    print(oracle_matrix)
    qc_oracle = QuantumCircuit(n_qubits)
    qc_oracle.append(UnitaryGate(oracle_matrix), range(n_qubits))
    #ut.show_figure(qc_oracle.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'}))
    ut.show_figure(qc_oracle.decompose().decompose().decompose().draw(output='mpl', style={'backgroundcolor': '#EEEEEE'}))
    #dag = circuit_to_dag(qc_oracle.decompose().decompose())
    #dag_drawer(dag)
    #ut.show_figure(qc_oracle.squ(oracle_matrix).draw(output='mpl', style={'backgroundcolor': '#EEEEEE'}))
    return qc_oracle, marked_indices


def create_oracle_circ_Toffoli(distances, threshold, n_qubits):
    """
    Create circuit for oracle - for one solution --> I - 2*|i0><i0|
    - decomposed to Generalized Toffoli Gate (n control qubits, 1 target qubits in state |-> and 2 symetrical gates on ith qubit, if ith binary digit of i0 is = 0)
        Args:
            distances: numpy.ndarray of shape (num_samples,)
                - Distances to be minimized
            threshold: int
            n_qubits: int
                - number of qubits oracle circuit is applied on.
        Returns:
            qc_oracle: Circuit()
                - oracle circuit
            marked_indices: int
                - number of indices which are marked as lower than threshold
    """
    # find a solution
    solution_bin = np.binary_repr(np.argmin(distances), width=n_qubits)
    solution_bin_reverse = solution_bin[::-1]
    index_0 = [pos for pos, char in enumerate(solution_bin_reverse) if char == '0']
    
    qc_oracle = Circuit(n_qubits+1) # for target qubit
    qc_oracle.add(gates.X(n_qubits))
    qc_oracle.add(gates.H(n_qubits))
    for q in index_0:
        qc_oracle.add(gates.X(q))
    qc_oracle.add(gates.X(n_qubits).controlled_by(index_0[0], index_0[1]))
    print(qc_oracle.draw())
    return qc_oracle
    