from math import log2, ceil
from random import random
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate
from .cmp import CMP
X = XGate()
Z = ZGate()

# type is removed as no Rfi needed now 
def CondMAJ(N , M,  k, array = None, as_circ = False):
    """Implements the conditional majority operation
       for a distribution of size M or a given array 
    Args : N - the size of the array 
           M - the size of the distribution 
           k - the number of copies we have 
           type - whether Rfi register is needed or 
                  not 
                  
    Returns : A quantum circuit implementing the 
              conditional majority operation   
    """
    # to test the circuit
    
    value_size = ceil(log2(M))
    encode_size = ceil(log2(k))
    
    circ = QuantumCircuit(name = 'CondMAJ')
    
    flip_regs = []
    
    for i in range(k):
        flip_reg = QuantumRegister(size = 1, name = f'flip_{i}')
        flip_regs.append(flip_reg)
        
    for flip_reg in flip_regs:
        circ.add_register(flip_reg) 
        
    # cache the value reg
    value_reg = QuantumRegister(size = value_size, name = 'value')
    circ.add_register(value_reg)
        
    # we also need an encoding of k/2 and a register for encoding the 
    # count of a particular xi 
    
    encoding_register = QuantumRegister(size = encode_size, name = 'value_encoding')
    k_register = QuantumRegister(size = encode_size, name = 'k_encoding')
    Ancilla = QuantumRegister(size = encode_size, name = 'ancilla')
    result_qubit = QuantumRegister(size = 1, name = 'result_cmaj')
    
    circ.add_register(encoding_register)
    circ.add_register(k_register)
    circ.add_register(Ancilla)
    circ.add_register(result_qubit)
    
    # get a CMP first, which is directly appended 
    cmp_circ = CMP(encode_size)
    
    # encode k_register 
    threshold = ceil(k/2)
    bin_rep = bin(threshold)[2:].zfill(encode_size)[::-1]
    
    # add this to k_register 
    for i,bit in enumerate(bin_rep):
        if bit == '1':
            circ.x(k_register[i])
    
    if as_circ:
        circ.barrier()
    
    
    def get_bin_encoder(value):
        rflip = QuantumRegister(size = 1)
        r1 = QuantumRegister(size = value_size)
        encoding = QuantumRegister(size = encode_size)
        
        circ = QuantumCircuit(rflip, r1, encoding)
        
        # make the ctrl state 
        ctrl_state = '1'*(encode_size-1)  + bin(value)[2:].zfill(value_size) + '1'
        
        # make the number of ctrl qubits
        ctrl_count = encode_size + value_size
        
        # make the circuit
        for i in range(encode_size):
            gate = X.control(num_ctrl_qubits = ctrl_count, ctrl_state = ctrl_state[i:])
            ctrl_count-=1 
            circ.compose(gate, inplace = True)
            
        return circ 
    
    # well if array is not None we can just encode for the 
    # unique elements of the array 
    if array is not None:
        elements = set(array)
    else:
        elements = list(range(M))
        
    qubits_val = list(range(k, k + value_size)) + list(range(k+value_size, k+value_size+encode_size))
    qubits_cmaj = [flip[:][0] for flip in flip_regs] + value_reg[:] + encoding_register[:]
    
    # make the circuit
    for value in elements:
        bin_encoding = get_bin_encoder(value)
        
        # make the encoder for the element 
        curr_circ = QuantumCircuit(value_size + k + encode_size)
        for i in range(k):
            qubit_idx = [i] + qubits_val
            curr_circ.compose(bin_encoding, qubits = qubit_idx , inplace = True)
        
        inv_circ = curr_circ.inverse()
        
        # add the encoder to the circuit
        circ.compose(curr_circ, qubits = qubits_cmaj, inplace = True) 
        
        # add the cmp operation
        circ.compose(cmp_circ, qubits = encoding_register[:] + k_register[:] + Ancilla[:] + [result_qubit[0]], inplace = True)
        circ.x(result_qubit[0])
        
        # add the inverse of the encoding you added
        circ.compose(inv_circ, qubits = qubits_cmaj, inplace = True)
        
        if as_circ:
            circ.barrier()
            
    if not as_circ:
        circ = circ.to_gate()
        circ.name = 'CMAJ'
    
    return circ 