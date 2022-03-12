# type specs
from ctypes import Union
from typing import List, Optional, Set
from builtins import int

# subroutines
from ..subroutines.cmaj import CondMAJ
from ..subroutines.cmp import CMP 
from ..subroutines.hdq import HDq
from ..subroutines.matrices import AMatrix 

# get the qae algorithm
from .qae import QAE

from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister

# helpers 
from math import ceil, pi, log, log2

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
    def threshold(self,val):
        self._check_param("Threshold", val)
        self._threshold = val  
         
    @property 
    def error(self):
        return self._error 
    
    @error.setter
    def error(self, val):
        self._check_param("Error",val)
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
    
    def _check_param(self,param, value):

        if not isinstance(value, float):
            raise TypeError(f"{param} must be a float value in range (0,1)")
        
        if value <= 0 or value >= 1:
            raise ValueError(f"{param} must lie between (0,1)")
            
            
    # main methods
    def encode(self, 
               arr : List, 
               M : int,
               N : Optional[int] = None,  
        ):
        
        # check arr
        if any( (not isinstance(element, int) or element < 0) for element in arr):
            raise Exception("Elements of array must be integers >= 0")

        # check N
        if N is None:
            N = len(arr)
            
        if int(log2(N)) != log2(N):
            raise ValueError("N must be a power of 2 for the correct functioning of algorithm")
        
        self._N = N 
        self._M = M 
        self._array = arr 
        
        self._has_encoding = True
        
        #handle ancilla, as N is now available
        size_anc = max([ceil(log2(self._N)) + ceil(log2(self._M))+ 1, 
                       self.precision + 1,
                       ceil(log2(self._K)) + 1])
        
        self._Ancilla = QuantumRegister(size = size_anc, name = 'ancilla')
        
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
        
    def get_circuit(self):
        if self._circuit is None:
            raise Exception("Algorithm needs at least one array encoding for the generation of the circuit")
        
        return self._circuit
        
    def run(self):
        pass 
    
    
    # helper methods
    def _setup_registers(self):
        
        # get the num of copies to use
        const = ((8 / pi**2) - 0.5)**(-2)
        self._K = ceil(const * log(1/(self.threshold * self.error)))
        logk = ceil(log2(self._K))
        print(f"K is {self._K}")
        # setup k precision and flip resgiters 
        prec_regs = {}
        
        for i in range(self._K):
            prec_regs[i] = (
                QuantumRegister(size = self.precision, name = f'prec_{i}'),
                QuantumRegister(size = 1, name = f'flip_{i}')
            )
            
        self._Prec_Regs = prec_regs
        
        # set up a global threshold register 
        self._Rthres = QuantumRegister(size = self.precision, name = 'thres_encoded')
        
        # registers for encoding the number of flip registers 
        # in the k copies
        self._Rt1 = QuantumRegister(size = logk, name = 'encoding')
        self._Rt2 = None # only for the Rfi encoding
        
        # setup k/2 encoding register and the result register 
        self._RK_half = QuantumRegister(size = logk, name = 'half_k')
        self._Rout = QuantumRegister(size = 1, name = 'output')
        
        
    def _setup_routines(self):
        # setup the CMP and HDq routines 
        self._HDq = HDq(self.precision)
        self._CMP = CMP(self.precision)
    
    
    def _setup_array_registers(self):
        size_n = ceil(log2(self._N))
        size_m = ceil(log2(self._M))
        
        self._R1 = QuantumRegister(size = size_n + size_m, name = "r1")
        self._R2 = QuantumRegister(size = size_n + size_m, name = "r2")
    
    def _setup_QRAM(self):
        self._oracle = AMatrix(self._array, self._N, self._M)
    
    def _make_QAE_operator(self):
        self._QAE = QAE(self._N, self._M, self._oracle, self.precision)
    
    def _make_cond_MAJ(self):
        self._Cond_MAJ = CondMAJ(self._N, self._M, self._K, self._array)
    
    def _define_circuit(self):
        """Create the empty circuit skeleton for the 
           algorithm
        """
        circuit = QuantumCircuit(
            name = 'hdist_circuit'
        )
         # add the qram and ancilla regs
        circuit.add_register(self._R1)
        circuit.add_register(self._R2)
        
        # add precision registers
        for reg_pair in self._Prec_Regs.values():
            prec , flip = reg_pair[0], reg_pair[1]
            circuit.add_register(prec)
            circuit.add_register(flip)
            
        # cached registers 
        circuit.add_register(self._Rthres)
        circuit.add_register(self._Ancilla)
            
        # add the Rt1 register and k/2 encoding register
        circuit.add_register(self._Rt1)
        
        if self._Rt2:
            circuit.add_register(self._Rt2)
        
        circuit.add_register(self._RK_half)
        circuit.add_register(self._Rout)
        
        
        self._circuit = circuit
        
        
    def _update_params(self):
        pass 
    
    def _construct_circuit(self):
        """Construct the High Dist circuit with 
           the components. 
           NOTE: the purpose of this method is to 
                 just construct, not optimize
        """
        
        # add the QRAM
        self._circuit.compose(self._oracle, self._R1[:], inplace = True)
        
        # add the encoding for threshold in self._Rthres
        def encode():
            pass 
        
        # add HDq on Rthres 
        self._circuit.compose(self._HDq, self._Rthres[:] + self._Ancilla[:self.precision + 1], inplace = True)
        
        qae_range = self._R1[:] + self._R2[:] + self._Ancilla[:self._R1.size+1]
        
        # start the qae operations 
        for i in range(self._K):
            prec_reg, flip_reg= self._Prec_Regs[i][0], self._Prec_Regs[i][1]
            
            # add qae 
            qae_bits = prec_reg[:] + qae_range
            hdq_bits = prec_reg[:] + self._Ancilla[:self.precision + 1]
            self._circuit.compose(self._QAE, qubits = qae_bits, inplace = True)
            
            self._circuit.compose(self._HDq, qubits = hdq_bits , inplace = True)
            
            # add cmp
            cmp_bits = prec_reg[:] + self._Rthres[:] + self._Ancilla[:self.precision] + flip_reg[:]
            self._circuit.compose(self._CMP, qubits = cmp_bits, inplace = True )
            
            # invert HDq
            self._circuit.compose(self._HDq, qubits = hdq_bits, inplace = True)
            
        # add the conditional MAJ 
        # get the flip precision register 
        precision_bits = [reg[1][0] for reg in self._Prec_Regs.values()]
        
        # get the total bits in the cond maj operation 
        value_size = ceil(log2(self._M))
        index_size = ceil(log2(self._N))
        
        # note : cmaj is not applied on whole register but just on the value half 
        c_maj_bits = precision_bits + self._R1[index_size: index_size+value_size] + self._Rt1[:] + self._RK_half[:] + self._Ancilla[:len(self._Rt1)] + self._Rout[:]
        self._circuit.compose(self._Cond_MAJ, qubits = c_maj_bits, inplace =True)
        
    # made circuit but bug in drawing ? 
    
    # qaa is left 
    
    
# Uncomment to test

h = HighDist(0.7, 0.4, 4)
h.encode([1,2,3,4], 7)
# print(h._QAE.decompose().draw())
h._construct_circuit()
h.get_circuit().draw('mpl', scale = 0.5, filename = '../../Circuits/imgs/high-dist.png') # correct definition