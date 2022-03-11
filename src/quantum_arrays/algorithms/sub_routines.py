from math import log2, ceil
from random import random
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate
import random 

X = XGate()
Z = ZGate()

# SUB-ROUTINES
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
    circ = QuantumCircuit(R1, Ancilla, R2)
    
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
        
    return circ


def EQ(N, as_circ = False):
    """
     Args
       N : size of the input register

     Returns : A QuantumCircuit which 
              implements the EQ operation,
              to be applied on the 
              circuit
    """
    R1 = QuantumRegister(name = 'r1', size = N)
    R2 = QuantumRegister(name = 'r2', size = N)
    Ancilla = QuantumRegister(name = 'ancilla', size = N + 1)
    
    circ = QuantumCircuit(R1, R2, Ancilla, name = 'EQ')
    
    # init dummy as |-> 
    circ.x(Ancilla[N])
    circ.h(Ancilla[N])
    
    for i in range(N):
        circ.cx(R1[i],Ancilla[i])
        circ.cx(R2[i],Ancilla[i])
        if as_circ:
            circ.barrier()
    gate = X.control(num_ctrl_qubits = N, ctrl_state = '0'*N)
    
    circ.compose(gate, qubits = Ancilla, inplace = True)
    if as_circ:
        circ.barrier()
        
    # uncompute 
    for i in range(N-1,-1,-1):
        circ.cx(R2[i],Ancilla[i])
        circ.cx(R1[i],Ancilla[i])
        if as_circ:
            circ.barrier()
    
    circ.h(Ancilla[N])
    circ.x(Ancilla[N])
    
    if not as_circ:
        circ = circ.to_gate()
        
    return circ

def CondMAJ(N , M,  k, type = None, array = None):
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
    
    # cache the value reg
    value_reg = QuantumRegister(size = value_size, name = 'value')
    circ.add_register(value_reg)
    
    for i in range(k):
        flip_reg = QuantumRegister(size = 1, name = f'flip_{i}')
        circ.add_register(flip_reg) 
        
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
            
    circ.barrier()
    
    
    def get_bin_encoder(value):
        r1 = QuantumRegister(size = value_size)
        rflip = QuantumRegister(size = 1)
        encoding = QuantumRegister(size = encode_size)
        
        circ = QuantumCircuit(r1, rflip, encoding)
        
        # make the ctrl state 
        ctrl_state = '1'*(encode_size-1) + '1' + bin(value)[2:].zfill(value_size)
        
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

    value_bits = list(range(value_size))
    encode_qubits = list(range((value_size+k),(value_size+k) + encode_size))
    
    # make the circuit
    for value in elements:
        bin_encoding = get_bin_encoder(value)
        
        # add the encoder to the circuit
        curr_circ = QuantumCircuit(value_size + k + encode_size)
        for i in range(k):
            curr_circ.compose(bin_encoding, qubits = value_bits + [i+value_size] + encode_qubits, inplace = True)
        
        inv_circ = curr_circ.inverse()
        
        circ.compose(curr_circ, qubits = list(range(value_size+k)) + encoding_register[:], inplace = True) 
        # add the cmp operation
        circ.compose(cmp_circ, qubits = encoding_register[:] + k_register[:] + Ancilla[:] + [result_qubit[0]], inplace = True)
        circ.x(result_qubit[0])
        # add the inverse of the encoding you added
        circ.compose(inv_circ, qubits = list(range(value_size+k)) + encoding_register[:], inplace = True)
        
        circ.barrier()
    
    return circ 


# ORACLES
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
    
def test_A(n,m):
    l = []
    for i in range(2**n):
        l.append(random.randint(1,(2**m)-1))

    print("arr is :",l)
    return AMatrix(l, 2**n, 2**m)
 

'''Testing space'''

# """Raw circuits"""
# print("CMP 5 :",CMP(5).draw())
# print("HDq 6 :",HDq(6).draw())
# print("EQ 4 :",EQ(4).draw())
# print("Cond MAJ operation for 8 element array sampled from size 4 distribution, with k = 4 :",CondMAJ(8,4,6).draw())

# a = test_A(3,3)

# print("All Zero flip 5 : ",FlipAllZero(5).draw())
# print("Q matrix :", QMatrix(6, a).decompose().draw())
