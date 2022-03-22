from random import random
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import XGate, ZGate

X = XGate()
Z = ZGate()


def EQ(N, as_circ=False):
    """
    Args
      N : size of the input register

    Returns : A QuantumCircuit which
             implements the EQ operation,
             to be applied on the
             circuit
    """
    R1 = QuantumRegister(name="r1", size=N)
    R2 = QuantumRegister(name="r2", size=N)
    Ancilla = QuantumRegister(name="ancilla", size=N + 1)

    circ = QuantumCircuit(R1, R2, Ancilla, name="EQ")

    # init dummy as |->
    circ.x(Ancilla[N])
    circ.h(Ancilla[N])

    for i in range(N):
        circ.cx(R1[i], Ancilla[i])
        circ.cx(R2[i], Ancilla[i])
        if as_circ:
            circ.barrier()
    gate = X.control(num_ctrl_qubits=N, ctrl_state="0" * N)

    circ.compose(gate, qubits=Ancilla, inplace=True)
    if as_circ:
        circ.barrier()

    # uncompute
    for i in range(N - 1, -1, -1):
        circ.cx(R2[i], Ancilla[i])
        circ.cx(R1[i], Ancilla[i])
        if as_circ:
            circ.barrier()

    circ.h(Ancilla[N])
    circ.x(Ancilla[N])

    if not as_circ:
        circ = circ.to_gate()
        circ.name = "EQ"

    return circ
