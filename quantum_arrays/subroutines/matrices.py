from math import log2, ceil
from random import random
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate

from .eq import EQ

X = XGate()
Z = ZGate()


def AMatrix(arr, N, M, as_circ = False):
    """Makes the oracle for the algorithm 
    Args : arr - the array to encode (list / iter)
           N   - size of the array (int) (2^k type)
           M   - the max element of the  (int)
                 distribution"""         
    index = 0 
    index_size = ceil(log2(N))
    value_size = ceil(log2(M))
    
    index_reg = QuantumRegister(name = "index", size = index_size)
    value_reg = QuantumRegister(name = "value", size = value_size)
    
    circ = QuantumCircuit(index_reg, value_reg, name = 'oracle')
    circ_cache = {}
    
    def get_elem_circ(elem, idx):
        # analyse this element 
        if elem in circ_cache:
            elem_circ = circ_cache[elem]
        else:
            bin_rep = bin(elem)[2:]
            elem_circ = QuantumCircuit(value_size, name = f'x_{idx}')
            bin_rep = bin_rep[::-1]
            
            for i,bit in enumerate(bin_rep):
                if bit == '1':
                    elem_circ.x(i)
            circ_cache[elem] = elem_circ
            

        elem_circ = elem_circ.control(num_ctrl_qubits = index_size, ctrl_state = idx)
        
        return elem_circ
    
    if hasattr(arr,'__next__'):
        it = arr 
        while it.next():
            elem = it 
            circuit = get_elem_circ(elem, index)
            circuit.name = f'x_{index}'
            circ.compose(circuit,  inplace = True) 
            index+=1             
    else:
        for elem in arr:
            circuit = get_elem_circ(elem, index)
            circuit.name = f'x_{index}'
            circ.compose(circuit, inplace = True)
            index+=1
    
    del circ_cache
    
    if not as_circ:
        circ = circ.to_gate()

    return circ

def FlipAllZero(N):
    """
    Implements the oracle to flip the 
    sign about the all |0> state in 
    the grover circuit
    
    Args:
        N : int - size of the register
        
    Returns:
        QuantumCircuit of size (N+1) where the 
        ancilla qubit is used for the flipping 
        of sign"""
    zero_reg = QuantumRegister(size = N, name = f'flip0_{N}')
    ancilla = QuantumRegister(size = 1)
    
    circ = QuantumCircuit(zero_reg, ancilla, name = 'zero_flip')
    
    circ.x(ancilla)
    cz = Z.control(num_ctrl_qubits = N, ctrl_state = '0'*N)
    circ.compose(cz, inplace = True)
    circ.x(ancilla)
    return circ.to_gate()
        
def QMatrix(N, oracle, disp = False):
    """Consructs the Q matrix required for 
       the QAE routine to be used in the 
       circuit
       
       Args:
            N : int - the size of the R1, R2 
                      registers 
            oracle : QuantumCircuit - the 
                      circuit implementing the 
                      A matrix 
                      
        Returns:
            QuantumCircuit of size (N + N + N + 1) 
            qubits to be appended to the circuit 
            """
    inv_oracle = oracle.inverse()
    
    R1 = QuantumRegister(size = N, name = 'r1')
    R2 = QuantumRegister(size = N, name = 'r2')
    Ancilla = QuantumRegister(size = N + 1, name = 'ancilla')
    
    circ = QuantumCircuit(R1, R2, Ancilla, name = 'Q')
    
    # make the oracle
    circ.compose(EQ(N), inplace = True)
    if disp:
        circ.barrier()
    
    # make the diffuser
    circ.compose(inv_oracle, qubits = R2[:], inplace = True)
    if disp:
        circ.barrier()
    circ.compose(FlipAllZero(N), qubits = R2[:] + [Ancilla[N]], inplace = True)
    if disp:
        circ.barrier()
    
    circ.compose(oracle, qubits = R2[:], inplace = True)
    
    return circ 
    