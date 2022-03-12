from random import random
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate

X = XGate()

def CMP(N, as_circ = False):
    """
    Args- 
    N : size of the R1 register on which 
        CMP is applied 
        
    Returns :
    qiskit.circuit.Instruction implementing
    the N-qubit CMP operation"""
    R1 = QuantumRegister(size = N, name = "r1")
    R2 = QuantumRegister(size = N, name = "r2")
    Ancilla = QuantumRegister(size = N, name = "ancilla")
    R3 = QuantumRegister(size = 1,name = "out")
    
    circ = QuantumCircuit(R1, R2, Ancilla, R3, name = 'CMP')
    
    # init R3 as |0>
    # circ.reset(R3[0])
    circ.x(R3[0])
    
    for i in range(N):
        id1 = N - i - 1 
        id2 = N - i - 1 
        gate = X.control(num_ctrl_qubits = i + 2, ctrl_state = '0'*(i+1) + '1')
        
        anc_qubits = [Ancilla[j] for j in range(i)]
        if len(anc_qubits)!=0:
            # print("Num of ancilla control :", len(anc_qubits))
            circ.compose(gate, qubits = [R1[id1], R2[ id2]] + anc_qubits + [Ancilla[N-1]], inplace = True, )
        else:
            circ.compose(gate, qubits = [R1[id1], R2[ id2], Ancilla[N-1]], inplace = True, )
            
        if i != N-1:
            circ.cx(R1[id1], R2[id2])
            circ.cx(R2[id1], Ancilla[i])
            circ.cx(R1[id1], R2[id2])
            
        if as_circ:
            circ.barrier()
        
        # print("Current :",circ.draw())
        
    inv_circ = circ.inverse()
    
    # add final flip 
    circ.cx(Ancilla[N-1], R3[0])
    
    circ = circ.compose(inv_circ)
    
    # remove the effect of last X
    circ.x(R3[0])
    
    if not as_circ:
        circ.to_gate()
    return circ 
