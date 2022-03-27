# first we need a circuit for the state preparation

# essentially the whole circuit
from qiskit.circuit.library import GroverOperator
from qiskit import QuantumCircuit


def good_state(bin_string):
    # qiskit uses little endian, that means
    # if meas output has 1 as the starting bit,
    # then it will be a good state
    bit = bin_string[0]
    return bit == "1"


def QAA(state_prep, iterations, disp=False):
    """Args :
    state_prep : the A matrix, for state preparation
    iterations : the number of iterations to use
                 in the amplification task"""

    # make an oracle which marks the state as
    # -1 when the last qubit is 1 as that is the good state here

    size = len(state_prep.qubits)
    oracle = QuantumCircuit(size, name="marking_oracle")
    oracle.z(size - 1)  # simple z marker

    circ = QuantumCircuit(size, name="qaa")

    grover_op = GroverOperator(
        oracle, state_preparation=state_prep, insert_barriers=disp
    )

    if disp:
        print("Grover operator is :", grover_op.decompose().draw())

    for _ in range(iterations):
        circ.compose(grover_op, inplace=True)

    # this function only provides the QAA routine
    # for your circuit
    circ.name = "QAA"
    return circ


# qc = QuantumCircuit(3, 1)
# qc.measure(-1, 0)
# qc.x(0)
# qc.h([1])
# qc.barrier()

# # simple 4 qubit circuit
# iters = 2
# # backend = Aer.get_backend("qasm_simulator")
# circ = QAA(qc, iters)

# qc.compose(circ, inplace=True)
# print(qc.draw())
