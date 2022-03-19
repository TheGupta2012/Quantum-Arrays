from random import random
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate

X = XGate()

 
def HDq(N, as_circ = False):
    """
    Args
       N : size of the input register

    Returns : A QuantumCircuit which
              implements the HDq operation
              ,to be applied in the 
              circuit"""
              
    R1 = QuantumRegister(size = N, name = 'r1')
    Ancilla = QuantumRegister(size = N, name = 'ancilla')
    R2 = QuantumRegister(size = 1, name = 'out')
    
    # HDq circuit 
    circ = QuantumCircuit(R1, Ancilla, R2, name = 'HDq')
    
    if as_circ:
        circ.reset(Ancilla)
        circ.reset(R2)
    
    for i in range(N):
        circ.cx(R1[i], Ancilla[i])
    
    if as_circ: 
        circ.barrier()
    
    # little endian
    circ.cx(R1[N-1], R2[0])
    circ.cx(R2[0], Ancilla[N-1])
    
    if as_circ:
        circ.barrier()
        
    controls = []
    for i in range(N):
        controls.append(Ancilla[i])
        gate = X.control(num_ctrl_qubits = i + 2 , ctrl_state = '1' + '0'*(i+1))
        
        for j in range(1, N - i - 1):
            circ = circ.compose(gate, qubits = [R2[0]] + controls[:i+1] + [Ancilla[N-1-j]])
    
    if as_circ:
        circ.barrier()
    
    # handle the all zero case
    gate = X.control(num_ctrl_qubits = N, ctrl_state = '0'*N)
    circ = circ.compose(gate)
    
    # uncompute
    circ.cx(R1[N-1], R2[0])
    
    #swap state
    circ.swap(R1, Ancilla)
    
    if not as_circ:
        circ = circ.to_gate()
        circ.name = 'HDq'
        
    return circ

