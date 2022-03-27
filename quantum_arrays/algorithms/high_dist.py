# type specs
import random
from typing import List, Optional
from builtins import int

# subroutines
from ..subroutines.cmaj import CondMAJ
from ..subroutines.cmp import CMP
from ..subroutines.hdq import HDq
from ..subroutines.matrices import AMatrix

# get the qae algorithm
from .qae import QAE

# get the qaa algorithm
from .qaa import QAA

# get imports from qiskit
from qiskit import ClassicalRegister, QuantumCircuit, assemble, execute, transpile
from qiskit.tools.monitor import job_monitor
from qiskit.circuit import QuantumRegister


# helpers
import math


class HighDist:
    def __init__(self, threshold, error, precision) -> None:
        self.precision = precision
        self.threshold = threshold
        self.error = error
        self._has_encoding = False
        self._circuit = None

        # set up registers, not dependent on the array
        self._setup_registers()
        self._setup_routines()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, val):
        self._check_param("Threshold", val)
        self._threshold = val

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, val):
        self._check_param("Error", val)
        self._error = val

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, val):
        if not isinstance(val, int):
            raise TypeError("Precision must be an int > 0")

        if val <= 0:
            raise ValueError("Precision must be atleast 1")
        self._precision = val

    def _check_param(self, param, value):

        if not isinstance(value, float):
            raise TypeError(f"{param} must be a float value in range (0,1)")

        if value <= 0 or value >= 1:
            raise ValueError(f"{param} must lie between (0,1)")

    # main methods
    def encode(
        self,
        arr: List,
        M: int,
        N: Optional[int] = None,
    ):

        # check arr
        if any((not isinstance(element, int) or element < 0) for element in arr):
            raise Exception("Elements of array must be integers >= 0")

        # check N
        if N is None:
            N = len(arr)

        if int(math.log2(N)) != math.log2(N):
            raise ValueError(
                "N must be a power of 2 for the correct functioning of algorithm"
            )

        self._N = N
        self._M = M
        self._array = arr

        # handle ancilla, as N is now available
        size_anc = max(
            [
                math.ceil(math.log2(self._N)) + math.ceil(math.log2(self._M)) + 1,
                self.precision + 1,
                math.ceil(math.log2(self._K)) + 1,
            ]
        )

        self._Ancilla = QuantumRegister(size=size_anc, name="ancilla")

        # define array regs
        self._setup_array_registers()

        # make oracle
        self._setup_QRAM()

        # make qae op
        self._make_QAE_operator()

        # make cond maj
        self._make_cond_MAJ()

        # make circuit skeleton
        self._define_circuit()

        self._has_encoding = True

    def get_circuit(self):
        if self._circuit is None:
            raise Exception(
                "Algorithm needs at least one array encoding for the generation of the circuit"
            )

        return self._circuit

    def run(
        self,
        backend,
        shots=1024,
        optimization_level=1,
        p_max_type=None,
        p_max_init=True,
        **kwargs,
    ):
        """Run the HighDist circuit for the given array encoding and params for the algo

           NOTE : HighDist algorithm is used in the Pmax circuit
                  Here, ONLY the
        Args:
            backend : backend to execute the circuit on
            shots (int, optional): Shots required for the measurement.
                                   Defaults to 1024.
            optimization_level (int, optional): Transpiler optimization level.
                                                Defaults to 1.
            p_max_type (str, optional): Flag to know whether the algorithm is running inside the
                                        Pmax algorithm or not
            p_max_init (bool, optional): Boolean to know whether the pmax algo is being run for
                                        the first time
        """

        if not self._has_encoding:
            raise Exception(
                "The circuit should have an encoding for it to be executed on a backend"
            )

        # 1. Clear out the circuit for a run
        self._define_circuit()

        # 2. Transpile the routines individually and then just
        #    construct the circuit with them
        components = ["_HDq", "_CMP", "_oracle", "_QAE", "_Cond_MAJ"]

        """For single transpilation only"""

        """If the precision does not change then the 
        QAE, HDq and CMP transpilations can only be done once 
        
        Also, for the oracle, since the array does not change 
        transpilation should only be done once 
        """

        # in additive type algo we only have a change in the threshold
        # and not the precision

        if kwargs["p_max_type"] == "additive" and not kwargs["p_max_init"]:
            components = ["Cond_MAJ"]

        for component in components:
            transpiled = transpile(
                self.__getattribute__(component),
                backend=backend,
                optimization_level=optimization_level,
            )
            self.__setattr__(component, transpiled)

        # 3. Construct the circuit now and execute
        self._construct_circuit()

        assembled_circ = assemble(self._circuit, backend=backend, shots=shots)
        job = execute(assembled_circ, backend=backend, shots=shots)

        job_monitor(job)

        # 4. Save the result of the executed circuit
        print("\n***HIGH DIST EXECUTED SUCCESSFULLY!***\n")
        self._result = {
            "job_result": job.result(),
            "counts": job.result().get_counts(),
            "algorithm_result": False,
        }
        counts = job.result().get_counts()

        if counts["1"] > counts["0"]:
            self._result["algorithm_result"] = True

    def get_results(self, verbose=False):
        """Get the results of the execution of HighDist algorithm

        Args:
            verbose (bool, optional): Verbosity of the result dictionary.
                                      Defaults to False.

        Returns:
            Dict: a dictionary containing the result of the execution
        """
        if not hasattr(self, "result"):
            raise AttributeError(
                "Result attribute not found. Please execute the circuit atleast once"
            )

        if verbose:
            result = self._result
        else:
            result = {
                "counts": self._result["counts"],
                "algorithm_result": self._result["algorithm_result"],
            }

        return result

    # helper methods
    def _setup_registers(self):

        # get the num of copies to use
        const = 0.5 * (((8 / math.pi ** 2) - 0.5) ** (-2))

        # Constant may be updated now
        """NOTE : """
        """Try to bound by using the MAX value of K which you can have  """
        self._K = math.ceil(const * math.log(1 / (self.threshold * self.error)))
        logk = math.ceil(math.log2(self._K))

        # setup k precision and flip resgiters
        prec_regs = {}

        for i in range(self._K):
            prec_regs[i] = (
                QuantumRegister(size=self.precision, name=f"prec_{i}"),
                QuantumRegister(size=1, name=f"flip_{i}"),
            )

        self._Prec_Regs = prec_regs

        # set up a global threshold register
        self._Rthres = QuantumRegister(size=self.precision, name="thres_encoded")

        # registers for encoding the number of flip registers
        # in the k copies
        self._Rt1 = QuantumRegister(size=logk, name="encoding")

        # setup k/2 encoding register and the result register
        self._RK_half = QuantumRegister(size=logk, name="half_k")
        self._Rout = QuantumRegister(size=1, name="output")

    def _setup_routines(self):
        # setup the CMP and HDq routines
        self._HDq = HDq(self.precision)
        self._HDq.name = "HDq"
        self._CMP = CMP(self.precision)
        self._CMP.name = "CMP"

    def _setup_array_registers(self):
        size_n = math.ceil(math.log2(self._N))
        size_m = math.ceil(math.log2(self._M))

        self._R1 = QuantumRegister(size=size_n + size_m, name="r1")
        self._R2 = QuantumRegister(size=size_n + size_m, name="r2")

    def _setup_QRAM(self):
        self._oracle = AMatrix(self._array, self._N, self._M)

    def _make_QAE_operator(self):
        self._QAE = QAE(self._N, self._M, self._oracle, self.precision)

    def _make_QAA_operator(self, circuit, iterations):
        return QAA(circuit, iterations, False)

    def _make_cond_MAJ(self):
        self._Cond_MAJ = CondMAJ(self._N, self._M, self._K, self._array)

    def _define_circuit(self):
        """Create the empty circuit skeleton for the
        algorithm
        """

        circuit = QuantumCircuit(name="hdist_circuit")

        # add precision registers
        for reg_pair in self._Prec_Regs.values():
            prec, flip = reg_pair[0], reg_pair[1]
            circuit.add_register(prec)
            circuit.add_register(flip)

        # add the qram and ancilla regs
        circuit.add_register(self._R1)
        circuit.add_register(self._R2)

        # cached registers
        circuit.add_register(self._Rthres)
        circuit.add_register(self._Ancilla)

        # add the Rt1 register and k/2 encoding register
        circuit.add_register(self._Rt1)
        circuit.add_register(self._RK_half)

        circuit.add_register(self._Rout)

        self._circuit = circuit

    def _encode_threshold(self):
        """Encode the threshold in the Rthres register in the circuit"""
        sin_inv = math.asin(math.sqrt(self.threshold - 2 ** (-self.precision - 3)))
        constant = math.floor(sin_inv * (2 ** (self.precision) / math.pi))

        bin_rep = bin(constant)[2:].zfill(self.precision)

        for i, bit in enumerate(bin_rep[::-1]):
            if bit == "1":
                self._circuit.x(self._Rthres[i])

    def _construct_circuit(self):
        """Construct the High Dist circuit with
        the components.
        NOTE: the purpose of this method is to
              just construct, not optimize
        """

        # add the QRAM
        self._circuit.compose(self._oracle, self._R1[:], inplace=True)
        self._circuit.barrier()

        # add the encoding for threshold in self._Rthres
        self._encode_threshold()

        qae_range = self._R1[:] + self._R2[:] + self._Ancilla[: self._R1.size + 1]
        hdq_thres_bits = self._Rthres[:] + self._Ancilla[: self.precision + 1]

        # start the qae operations
        for i in range(self._K):
            prec_reg, flip_reg = self._Prec_Regs[i][0], self._Prec_Regs[i][1]

            hdq_prec_bits = prec_reg[:] + self._Ancilla[: self.precision + 1]

            # add qae
            qae_bits = prec_reg[:] + qae_range
            self._circuit.compose(self._QAE, qubits=qae_bits, inplace=True)
            self._circuit.barrier()

            # add the HDq circuit
            self._circuit.compose(self._HDq, qubits=hdq_thres_bits, inplace=True)
            self._circuit.compose(self._HDq, qubits=hdq_prec_bits, inplace=True)

            # add cmp
            cmp_bits = (
                prec_reg[:]
                + self._Rthres[:]
                + self._Ancilla[: self.precision]
                + flip_reg[:]
            )
            self._circuit.compose(self._CMP, qubits=cmp_bits, inplace=True)

            # invert HDq circuit
            self._circuit.compose(self._HDq, qubits=hdq_thres_bits, inplace=True)
            self._circuit.compose(self._HDq, qubits=hdq_prec_bits, inplace=True)

            self._circuit.barrier()

        # add the conditional MAJ
        # get the flip precision register
        precision_bits = [reg[1][0] for reg in self._Prec_Regs.values()]

        # get the total bits in the cond maj operation
        value_size = math.ceil(math.log2(self._M))
        index_size = math.ceil(math.log2(self._N))

        # NOTE : condition on the R1 register's values not R2

        # note : cmaj is not applied on whole register but just on the value half
        c_maj_bits = (
            precision_bits
            + self._R1[index_size : index_size + value_size]
            + self._Rt1[:]
            + self._RK_half[:]
            + self._Ancilla[: len(self._Rt1)]
            + self._Rout[:]
        )

        self._circuit.barrier()
        self._circuit.compose(self._Cond_MAJ, qubits=c_maj_bits, inplace=True)

        iterations = 2
        self._QAA = self._make_QAA_operator(self._circuit, iterations)
        self._circuit.compose(self._QAA, qubits=self._circuit.qubits, inplace=True)

        self._circuit.barrier()

        self._circuit.add_register(ClassicalRegister(1, name="cl_bit"))

        # measure the last qubit in the circuit
        self._circuit.measure(-1, 0)

    # qaa left
    # made circuit but bug in drawing ?

    # qaa is left
    def _update_params(self, threshold, precision=None):
        """Update the params of the HighDist
        algorithm to correctly run the Pmax
        circuit.

        Args:
             threshold : the new threshold of the algorithm
             precision : the new precision of the algorithm
        """

        self.threshold = threshold
        if precision is not None:
            self.precision = precision

        self._setup_registers()

        # array is same, no encoding required

        # we also need to have a check over the ancilla
        # as precision may have changed
        size_anc = max(
            [
                math.ceil(math.log2(self._N)) + math.ceil(math.log2(self._M)) + 1,
                self.precision + 1,
                math.ceil(math.log2(self._K)) + 1,
            ]
        )

        self._Ancilla = QuantumRegister(size=size_anc, name="ancilla")

        # define skeleton again
        self._define_circuit()

        # okay, if I even Do setup these registers, are they gonna be empty?
        # yes, they are gonna be because the operations happen on the circuit
        # and not a register


# Uncomment to test
h = HighDist(0.75, 0.33, 4)
list_num = []

for _ in range(2 ** 4):
    list_num.append(random.randint(1, 2 ** 4))

print(list_num)
h.encode(arr=list_num, M=2 ** 4 + 1)
h._construct_circuit()

circ = h.get_circuit()
circ.draw("text", filename="hdist.txt", scale=0.6)  # correct definition
# circ.draw('latex', filename = 'hdist.tex', scale = 0.7)
# manhattan = FakeManhattan()

# from qiskit import transpile
# from qiskit.providers.aer import AerSimulator

# # try to do more here...
# backend = AerSimulator(method="matrix_product_state")
# h.run(backend, 1024)

# print(h.get_results())
# circ_decomp = circ.decompose()
# print("Depth is :", circ_decomp.depth())

# # circ_decomp.draw('text', filename = 'hdist_decompose.txt', scale = 0.7)
