"""Defines an implementation of the Slater Determinant states,with a dictionary to represent the occupied states with the corresponding constants.
"""
# Defines an implementation of the Slater Determinant states,with a dictionary to represent the occupied states with the corresponding constants.
import numpy as np
from itertools import product
import copy
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool
import openfermion as of
import os

# Parallelized adding of sdstates, to be further improved
def reducer(a,b):
    return a+b
def parallel_reduce_partial(results):
    num_processes = min(len(results), os.cpu_count())

    # If there's only one or two results, no need for further parallel processing
    if num_processes <= 2:
        return reduce(reducer, results)

    with Pool(processes=num_processes) as pool:
        # Combine the results into pairs and reduce each pair
        # If the number of results is odd, one will be left out and added at the end
        paired_results = [(results[i], results[i+1]) for i in range(0, len(results)-1, 2)]
        reduced_pairs = pool.starmap(reducer, paired_results)

        # If there was an odd result out, add it to the list
        if len(results) % 2 != 0:
            reduced_pairs.append(results[-1])

        # Further reduce if necessary
        return parallel_reduce_partial(reduced_pairs)
    
def int_to_str(k, n_qubits):
    return str(bin(k))[2:][::-1] + "0" * (n_qubits - k.bit_length())

class sdstate:
    eps = 1e-8
    dic = {}
    n_qubit = 0
    def __init__(self, s = None, coeff = 1, n_qubit = 0, eps = 1e-8):
        self.dic = {}
        self.eps = eps
        if n_qubit:
            self.n_qubit = n_qubit
        if s:
            self.dic[s] = coeff
            if not self.n_qubit:
                self.n_qubit = len(str(bin(s)))- 2
            
    def norm(self):
#         Return the norm of the current state
        return np.sqrt(self @ self)

    def remove_zeros(self):
#         Remove zeros in the state
        self.dic = {k: coeff for k, coeff in self.dic.items() if coeff != 0}
        return None
    
    def normalize(self):
#         Normalize the current state
        n = self.norm()
        self.remove_zeros()
        for i in self.dic:
            self.dic[i] /= n
        return None
    
    
    def __add__(self, other):
        # Create a new instance with the updated number of qubits
        result = sdstate(n_qubit=max(self.n_qubit, other.n_qubit))
        # Copy the dictionary from self
        result.dic = self.dic.copy()

        # Iterate over other.dic to add or update the states in result.dic
        for s, coeff in other.dic.items():
            result.dic[s] = result.dic.get(s, 0) + coeff

        return result

    def __sub__(self, other):
        # Create a new instance with the updated number of qubits
        result = sdstate(n_qubit=max(self.n_qubit, other.n_qubit))
        # Copy the dictionary from self
        result.dic = self.dic.copy()
        # Iterate over other.dic to add or update the states in result.dic
        for s, coeff in other.dic.items():
            result.dic[s] = result.dic.get(s, 0) - coeff
        return result
    
    def __mul__(self, n):
#         Defines constant multiplication
        result = copy.deepcopy(self)
        for s in result.dic:
            result.dic[s] *= n
        return result
    
    def __rmul__(self, n):
        return self.__mul__(n)
    
    def __truediv__(self, n: int):
        return self.__mul__(1/n)
    
    def __matmul__(self, other):
        if isinstance(other, of.FermionOperator):
            return self.Hf_state(other)
        if self.n_qubit == 0 or other.n_qubit == 0:
            return 0
        count = 0
#         assert self.n_qubit == other.n_qubit, "qubit number mismatch"
        lis = list(set(list(self.dic.keys())) & set(list(other.dic.keys())))
        for s in lis:
            count += np.conjugate(self.dic[s]) * other.dic[s]
        return count 
    
    def __str__(self):
        self.remove_zeros()
        return str({int_to_str(k, self.n_qubit): self.dic[k] for k in self.dic})
    
    def exp(self, Hf):
        """Return the expectation value Hamiltonian on the current state with Hamiltonian in the form of either
        of.FermionOperator, 1e or 2e tensor, or a tuple of (1e, 2e) tensor
        """
        if isinstance(Hf, of.FermionOperator):
            return np.real(self @ self.Hf_state(Hf))
        elif isinstance(Hf, np.ndarray):
            return np.real(self @ self.tensor_state(Hf))
#         Handles a tuple of 1e and 2e tensor
        elif isinstance(Hf, tuple):
            return self.exp(Hf[0]) + self.exp(Hf[1])
        print("Invalid input of Hf")
        return -1
    
    def tensor_state(self, tbt):
        n = tbt.shape[0]
        assert len(tbt.shape) == 2 or len(tbt.shape) == 4, "Invalid tensor shape"
        re_state = sdstate(n_qubit = self.n_qubit)
        if len(tbt.shape) == 4:
            for p, q, r, s in product(range(n), repeat = 4):
                if tbt[p, q, r, s] != 0:
                    re_state += tbt[p, q, r, s] * sdstate.Epqrs(self, self.n_qubit - 1 - p, self.n_qubit - 1 -q, self.n_qubit - 1 - r, self.n_qubit - 1 -s)
        elif len(tbt.shape) == 2:
            for p, q in product(range(n), repeat = 2):
                if tbt[p,q] != 0:
                    re_state += tbt[p, q] * sdstate.Epq(self, self.n_qubit - 1 - p, self.n_qubit - 1 - q)
        return re_state
        
    def concatenate(self, st):
#         Return the direct product of two sdstates
        if len(self.dic) == 0:
            return st
        elif len(st.dic) == 0:
            return self
        n2 = st.n_qubit
        n = self.n_qubit + st.n_qubit
        tmp = sdstate(n_qubit = n)
        for s1 in self.dic:
            for s2 in st.dic:
                tmp += sdstate(s = s1 << n2 | s2, coeff = self.dic[s1] * st.dic[s2], n_qubit = n)
        return tmp
                
    def Epq(self, p, q):
        """
        Return the action of a_p^a_q on the current state.
        """
        tmp = sdstate(n_qubit = self.n_qubit)
        for n in self.dic:
            if actable_pq(n, p, q):
                t = n ^ (1 << p) ^ (1 << q)
                tmp += sdstate(t, self.dic[n] * (-1) ** parity_pq(n, p, q), n_qubit = self.n_qubit)
        return tmp
    
    def Epqrs(self, p, q, r, s):
#         To be changed or improved? Current implementation based on Epq
        """
        Return the action of a_p^a_q on the current state.
        """
        tmp = sdstate(n_qubit = self.n_qubit)
        for n in self.dic:
            if actable_pq(n, r, s):
                t = n ^ (1 << r) ^ (1 << s)
                if actable_pq(t, p, q):
                    k =  t ^ (1 << p) ^ (1 << q)
                    tmp += sdstate(k, self.dic[n] * (-1) ** (parity_pq(n, r, s) + parity_pq(k, p, q)), n_qubit = self.n_qubit)
        return tmp

    # Function to process a batch of terms
    def process_batch(self,batch):
        partial_state = sdstate(n_qubit=self.n_qubit)
        for t, coeff in batch:
            if coeff != 0:
                partial_state += self.op_state(t, coeff)
        return partial_state
    
    def Hf_state(self, Hf: of.FermionOperator, multiprocessing = False):
        """Apply a Hamiltonian in FermionOperator on the current state. multiprocessing can be used
        to parallelize the process of applying each Excitation operator in the Hamiltonian. The general
        cost is given by O(N^4M), for N as the qubit dimension and M as the size of the current state.
        """
        H = of.transforms.chemist_ordered(Hf)
        re_state = sdstate(n_qubit = self.n_qubit)
        if multiprocessing:
            # Convert FermionOperator terms to a list of tuples for easy batch processing
            terms_list = [(t, H.terms[t]) for t in H.terms]

            # Determine the optimal number of batches
            num_processes = min(len(terms_list), os.cpu_count())
            batch_size = len(terms_list) // num_processes

            # Create batches and distribute them across processes
            batches = [terms_list[i:i + batch_size] for i in range(0, len(terms_list), batch_size)]

            with Pool(processes=num_processes) as pool:
                results = pool.map(self.process_batch, batches)

            # Efficiently combine the results
            for partial_state in results:
                re_state += partial_state
        else:
            for t in H.terms:
                re_state += self.op_state(t, H.terms[t])
        return re_state

    def op_state(self, t, coef):
        if coef != 0:
            if len(t) == 4:
                return coef * self.Epqrs(self.n_qubit - 1 - t[0][0], self.n_qubit - 1 - t[1][0],
                                         self.n_qubit - 1 - t[2][0], self.n_qubit - 1 - t[3][0])
            elif len(t) == 2:
                return coef * self.Epq(self.n_qubit - 1 - t[0][0], self.n_qubit - 1 - t[1][0])
            elif len(t) == 0:
                return coef * self
        return sdstate(n_qubit = self.n_qubit)
    
    def to_vec(self):
        vec = np.zeros(2 ** self.n_qubit)
        for i in self.dic:
            vec[i] = self.dic[i]
        return vec
    
def parity_pq(number: int, a, b):
    """Count the number of electrons between p and q bits (p+1, p+2, ... ,q-1), 
    return a binary number representing the parity of the substring in the binary representation of number
    """
    if abs(a - b) < 2:
        return 0
    p = min(a,b)
    q = max(a,b)
    # Create a mask with 1s between p/
    mask = ((1 << q) - 1)

    # Apply the mask to truncate the first q bits, and drop the last p bits.
    result = (number & mask ) >> (p + 1)
    # Compute the parity
    parity = 0
    while result:
        parity ^= 1
        result &= result - 1  # Drops the lowest set bit
    return parity

def actable_pq(n, p, q):
    """
    Determines if a_p^a_q annihilates the current state given by n, for n as an index in fock space
    """
    return (p == q and (n & 1 << q) != 0) or ((n & 1 << p) == 0 and (n & 1 << q) != 0)
    