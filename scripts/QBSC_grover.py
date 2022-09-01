import sys
#sys.path.append('/eos/home-e/epuljak/private/epuljak')
sys.path.append('../')
sys.path.append('../../')

from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.visualization.utils import _bloch_multivector_data
from qiskit import QuantumCircuit, execute, Aer, transpile
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, IBMQ, execute

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import h5py

import scripts.distance_calc as distc
import scripts.util as ut

def convert_to_FP(inputs, lat_dim):    
    # max_dist = math.ceil(max(inputs))
    # #max_dist = math.ceil(math.sqrt(4*lat_dim))
    # print(max_dist)
    # n_bits = int(math.log(max_dist+2)/math.log(2))+1
    # print(n_bits)
    # step = max_dist/2**n_bits
    # inputs_fp = [i/step for i in inputs]
    
    # from rig.type_casts import NumpyFloatToFixConverter, NumpyFixToFloatConverter
    # n_bits=8
    # converter = NumpyFloatToFixConverter(signed=False, n_bits=n_bits, n_frac=2)
    # inputs_fp = converter(inputs)
    
    n_frac=2
    n_bits=4
    min_value = 0.
    max_value = 2**n_bits - 1
    print(max_value)
    # Scale and cast to appropriate int types
    vals = inputs * 2.0 ** n_frac
    print(vals)
    # Saturate the values
    vals = np.clip(vals, min_value, max_value)

    inputs_fp = np.array(vals, copy=True)
    return inputs_fp, n_bits

def basis_encoding_ampl(inputs, n_qubits):
    amplitudes = np.zeros(2**n_qubits)
    for i in inputs:
        amplitudes[i] = 1./math.sqrt(len(inputs))
    print(amplitudes)
    return amplitudes

def basis_encoding_stateprep(n_qubits, amplitudes):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.initialize(amplitudes)
    qc.draw()
    state_vec = Statevector.from_instruction(qc)
    return qc, state_vec

def get_per_qubit_params(statevector, n_qubits):
    bloch_data = (_bloch_multivector_data(statevector))
    
    phi=[]; theta=[]
    x=[]; y=[]; z=[]
    for n in range(n_qubits):
        qubit_vec = bloch_data[n]
        x.append((qubit_vec[0]))
        y.append((qubit_vec[1]))
        z.append((qubit_vec[2]))
        if qubit_vec[0] == 0.0: phi.append(0.)
        else: phi.append(math.atan(qubit_vec[1]/qubit_vec[0]))
        theta.append(math.acos(qubit_vec[2]))
    return x, y, z, phi, theta     # because you can use RVGate to rotate using x, y, z


def UC_gate():
    qc = QuantumCircuit(4, name='UC')
    
    qc.x(1)
    qc.ccx(0, 1, 2)
    qc.x(0); qc.x(1)
    qc.ccx(0, 1, 3)
    qc.x(0)
    
    return qc.to_gate()

def UC_T_gate():
    qc = QuantumCircuit(4, name='UC_T')
    
    qc.x(0)
    qc.ccx(0, 1, 3)
    qc.x(0); qc.x(1)
    qc.ccx(0, 1, 2)
    qc.x(1)
    
    return qc.to_gate()

def Toffoli_reverse():
    qc = QuantumCircuit(3, name='TF_reverse')
    
    qc.x(0); qc.x(1)
    qc.ccx(0,1,2)
    qc.x(0); qc.x(1)
    
    return qc.to_gate()

def inversion_around_mean(n_qubits, list_qubits):
    qc = QuantumCircuit(n_qubits, name='IAM')
    controlled_qbits=[]
    for qubit in list_qubits:
        if qubit+5<n_qubits:
            controlled_qbits.append(qubit)
        qc.h(qubit) # apply H-gate
        qc.x(qubit) # apply X-gate

    qc.h(list_qubits[-1])
    qc.mct(controlled_qbits, list_qubits[-1])
    qc.h(list_qubits[-1])

    for qubit in list_qubits:
        qc.x(qubit) # apply H-gate
        qc.h(qubit) # apply X-gate
    
    return qc

def state_preparation(n_qubits, n_qubits_fp, threshold_bin, state_vec_b):
    
    _,_,_, phi_b, theta_b = get_per_qubit_params(state_vec_b, n_qubits_fp)
    
    qc = QuantumCircuit(n_qubits+2, n_qubits_fp, name='Q_Oracle') # 1: |->, 2: |0> ancilla
    
    i=0
    j_a=0; assign_a=False
    j_b=0; assign_b=True
    b_indices=[]
    while i < n_qubits:
        if not assign_a:
            if threshold_bin[j_a]=='1':
                qc.x(i)
            j_a+=1
            assign_a = True; assign_b = False
            i+=1
            continue
        if not assign_b:
            qc.r(theta_b[j_b], phi_b[j_b], qubit=i)
            j_b+=1
            b_indices.append(i)
            assign_b = True; assign_a = False
            i+=4
            continue
    # ancilla |->
    qc.x(n_qubits)
    qc.h(n_qubits)
    
    return qc, b_indices

def qbsc_oracle(threshold, state_vec_b, n_qubits_fp):
    
    n_qubits_qbsc = n_qubits_fp*2 + (3*(n_qubits_fp-1) + 2)
    
    # binary repr. of threshold
    thresh_bin = np.binary_repr(int(threshold), width=int(n_qubits_fp))
    thresh_bin = list(thresh_bin)
    
    qc, b_indices = state_preparation(n_qubits_qbsc, n_qubits_fp, thresh_bin, state_vec_b)
    
    # append UC gate
    uc = UC_gate()
    i=0
    while i+5 < n_qubits_qbsc:
        qc.append(uc, [i, i+1, i+2, i+3])
        i+=5
    qc.barrier()
    print("Added UC gate")
    
    # reverse Toffoli
    r_toffoli = Toffoli_reverse()
    i=2
    while i < n_qubits_qbsc:
        qc.append(r_toffoli, [i, i+1, i+2])
        if i+10 < n_qubits_qbsc:
            i+=5
        else: break
    qc.barrier()
    print('Added Reversed Toffoli')
    
    # change dominance to ith bit - if needed
    i = n_qubits_qbsc-1
    while i - 5 > 0:
        qc.ccx(i-1, i-4, i-6)
        qc.ccx(i, i-4, i-5)
        qc.barrier()
        i-=5
    print("Added change of dominance")
    
    # measure O2 - tells us a<b
    qc.cx(3, n_qubits_qbsc+1)
    
    # measure if O2=O1
    qc.x(2)
    qc.x(3)
    qc.ccx(2, 3, n_qubits_qbsc+1)
    qc.x(2)
    qc.x(3)
    qc.barrier()
    
    # flip ancilla
    qc.cx(n_qubits_qbsc+1, n_qubits_qbsc)
    qc.barrier()
    print("Measure o2, o2=o1 and flip ancilla")
    
    # ========= MIRROR ALL BACK - to return states to prior state (only ancilla needed to be flipped) ======
    # O2=O1 ?
    qc.x(2)
    qc.x(3)
    qc.ccx(2, 3, n_qubits_qbsc+1)
    qc.x(2)
    qc.x(3)
    # O2
    qc.cx(3, n_qubits_qbsc+1) # measure O2
    qc.barrier()
    print('Mirror 1')
    
    # change dominance back
    i = n_qubits_qbsc-1
    while i - 5> 0:
        qc.ccx(i, i-4, i-5)
        qc.ccx(i-1, i-4, i-6)
        qc.barrier()
        i-=5
    qc.barrier()
    print("Mirror 2")
    
    # reverse Toffoli
    r_toffoli = Toffoli_reverse()
    i=2
    while i < n_qubits_qbsc:
        qc.append(r_toffoli, [i, i+1, i+2])
        if i+10 < n_qubits_qbsc:
            i+=5
        else: break
    qc.barrier()
    print('Mirror 3')
    
    # append UC_T gate
    uc_t = UC_T_gate()
    i=0
    while i+5 < n_qubits_qbsc:
        qc.append(uc_t, [i, i+1, i+2, i+3])
        i+=5
    qc.barrier()
    print('Mirrow final')
    
    return qc, b_indices # return QBSC as oracle

def grover_algo(distances, lat_dim, n_shots=1024):
    k = len(distances) # number of clusters
    #dist_fp, n_qubits_fp = convert_to_FP(distances, lat_dim) # convert to fixed-point
    dist_fp = distances
    n_qubits_fp = 4
    print(dist_fp)
    # choose random threshold
    index_rand = np.random.choice(list(range(k))) # random index of reference state
    threshold = dist_fp[index_rand]
    
    # create state vector b - basis encoding for all states in dist_fp
    ampl_b = basis_encoding_ampl(dist_fp, n_qubits_fp)
    _, state_vec_b = basis_encoding_stateprep(n_qubits_fp, ampl_b)
    print(f'state b: {state_vec_b.to_dict()}')
    
    start_index = k
    min_found=False
    measured_indices=[]
    while start_index > 0:
        print(start_index)
        print(f'Current threshold: {threshold}')
        measured_indices.append(threshold)
        # create oracle
        qc, b_indices = qbsc_oracle(threshold, state_vec_b, n_qubits_fp)
        
        # add inversion around mean
        qc_inv_mean = inversion_around_mean(qc.num_qubits, b_indices)
        
        qc_total = qc + qc_inv_mean        
        
        qc_total.measure(b_indices, list(range(n_qubits_fp)))
        # ut.show_figure(qc.draw())
        
        #run oracle+inversion
        simulator = Aer.get_backend('qasm_simulator')
        circ = transpile(qc_total, simulator)

        # Run and get counts
        result = simulator.run(circ, shots=n_shots).result()
        counts = result.get_counts(circ)
        
        ut.show_figure(plot_histogram(counts))
        probs = counts.items()
        sorted_probs = dict(sorted(probs, key=lambda item: item[1], reverse=True))
        sorted_probs_keys = list(sorted_probs.keys())
        
        print(counts)
        print(sorted_probs_keys)
        
        if sorted_probs_keys[0] == np.binary_repr(threshold, width=n_qubits_fp):
            threshold = int(sorted_probs_keys[0],2)
            min_found=True
            #measured_indices.append(threshold)
            start_index-=1
        else:
            for i in range(start_index):
                if int(sorted_probs_keys[i],2) in dist_fp and int(sorted_probs_keys[i],2) not in measured_indices:
                    threshold = int(sorted_probs_keys[i],2)
            #measured_indices.append(threshold)
            start_index-=1
            
        if start_index == 0 or min_found:
            minimum = threshold
            print(f'Minimum found: {minimum}')
            break
        # get b state for next iteration
        inputs = []
        for i, key in enumerate(sorted_probs_keys):
            if int(key,2) not in measured_indices and int(key,2) in dist_fp:
                inputs.append(int(key,2))
            if i == start_index-1: break
        print(f'b state for next: {inputs}')
        ampl_b = basis_encoding_ampl(inputs, n_qubits_fp)
        _, state_vec_b = basis_encoding_stateprep(n_qubits_fp, ampl_b)
        print(state_vec_b.to_dict())
    return minimum, qc, qc_inv_mean, qc_total, counts
    #return qc_total, counts