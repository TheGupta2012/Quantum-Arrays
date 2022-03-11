from sub_routines import QMatrix,test_A
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.algorithms import AmplitudeEstimation, EstimationProblem 
from math import ceil, log2

def QAE(N, M, A, precision):
    """Returns a QAE circuit to be 
       attached to the main circuit 
       of the HighDist algorithm
       
       Args :
            N : size of the R1 and R2 registers
            A : oracle for the encoding of the 
                array / distribution
            precision : number of bits of precision     
            
        Returns : A QuantumCircuit implementing the 
                  QAE algorithm
    """
    size_reg = ceil(log2(N)) + ceil(log2(M))
    R1 = QuantumRegister(name = 'r1', size = size_reg)
    R2 = QuantumRegister(name = 'r2', size = size_reg)
    Ancilla = QuantumRegister(name = 'ancilla' , size = size_reg + 1)
    
    # circ = QuantumCircuit(R1, R2, Ancilla, Precision)

    # just make the Oracle A as A tensor I
    Oracle = QuantumCircuit(R1, R2, Ancilla, name = 'oracle')
    Oracle.compose(A, qubits = R2[:size_reg], inplace = True)
    
    GroverOp = QMatrix(size_reg,A)
    
    problem = EstimationProblem(
        state_preparation = Oracle, 
        grover_operator = GroverOp,
        objective_qubits = [0]
    )
    
    AmpEst = AmplitudeEstimation(
        num_eval_qubits = precision
    )
    
    circuit = AmpEst.construct_circuit(problem)
    circuit.name = 'qae_circ'

    return circuit

# uncomment to test

# a = test_A(2,2)
# q = QAE(4,4,a,3)

# print(q.decompose().draw())
# print("Classical :",q.num_clbits)
# print("Qubits :", q.num_qubits)
