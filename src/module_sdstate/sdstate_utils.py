"""Defines an implementation of the Slater Determinant states, with a dictionary to 
represent the occupied states with the corresponding constants. Currently encorporating 
with 1e and 2e tensor, and Hamiltoniana in FermionOperator in chemist notation.
"""
import math
import numpy as np
from itertools import product
import copy
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
    
class sdstate:
    eps = 1e-8
    dic = {}
    n_qubit = 0
    def __init__(self, s = None, coeff = 1, n_qubit = 0, dic = None, eps = 1e-8):
        self.dic = {}
        self.eps = eps
        if n_qubit:
            self.n_qubit = n_qubit
        if isinstance(s, int):
            self.dic[s] = coeff
            if not self.n_qubit:
                self.n_qubit = len(str(bin(s))) - 2
        if isinstance(dic, dict):
            self.dic = dic
            if not self.n_qubit:
                n = 0
                for s in self.dic:
                    n = max(n, len(str(bin(s))) - 2)
                self.n_qubit = n
            
    def norm(self):
#         Return the norm of the current state
        return np.sqrt(self @ self)

    def remove_zeros(self):
#         Remove zeros in the state, removing states smaller than self.eps
        self.dic = {k: coeff for k, coeff in self.dic.items() if np.linalg.norm(coeff) > self.eps}
        return None
    
    def normalize(self):
#         Normalize the current state
        n = self.norm()
        self.remove_zeros()
        for i in self.dic:
            self.dic[i] /= n
        return None
    
    def __eq__ (self, other):
        return self.dic == other.dic and self.n_qubit == other.n_qubit

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
        # Defines constant multiplication
        result = copy.deepcopy(self)
        for s in result.dic:
            result.dic[s] *= n
        return result
    
    def __rmul__(self, n):
        return self.__mul__(n)
    
    def __truediv__(self, n: float):
        return self.__mul__(1/n)
    
    def __matmul__(self, other):
        if isinstance(other, of.FermionOperator):
            return self.Hf_state(other)
        if self.n_qubit == 0 or other.n_qubit == 0:
            return 0
        return self.inner(other) 
    
    def __str__(self):
        self.remove_zeros()
        return str({int_to_str(k, self.n_qubit): self.dic[k] for k in self.dic})
    
    def inner(self, other):
        """
        Defines the inner product of the current state with another state
        """
        count = 0
        lis = list(set(list(self.dic.keys())) & set(list(other.dic.keys())))
        for s in lis:
            count += np.conjugate(self.dic[s]) * other.dic[s]
        return count 


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
                    re_state += tbt[p, q, r, s] * self.Epqrs(p,q,r,s)
        elif len(tbt.shape) == 2:
            for p, q in product(range(n), repeat = 2):
                if tbt[p,q] != 0:
                    re_state += tbt[p, q] * self.Epq(p,q)
        return re_state
        
    def concatenate(self, st):
        """
        Return the direct product of two sdstates.
        |SD> = |SD> \otimes |st>
        """
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
    
    def truncate(self, n):
#         Truncate the state, leaving only the n states of leading coefficients
        self.dic = dict(sorted(self.dic.items(), key=lambda item: abs(item[1]), reverse=True)[:n])
    
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
    
    def Hf_state(self, H: of.FermionOperator, multiprocessing = False):
        """Apply a Hamiltonian in FermionOperator on the current state. multiprocessing can be used
        to parallelize the process of applying each Excitation operator in the Hamiltonian. The general
        cost is given by O(N^4M), for N as the qubit dimension and M as the size of the current state.
        """
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
                return coef * self.Epqrs(t[0][0], t[1][0],
                                         t[2][0], t[3][0])
            elif len(t) == 2:
                return coef * self.Epq(t[0][0], t[1][0])
            elif len(t) == 0:
                return coef * self
        return sdstate(n_qubit = self.n_qubit)
    
    def to_vec(self):
        """
        Convert to np.ndarray or scipy.sparse
        """
        # dim = 2 ** self.n_qubit
        vec = np.zeros(2 ** self.n_qubit)
        for i in self.dic:
            vec[reverse_bits(i, self.n_qubit)] = self.dic[i]
        return vec

    def get_1RDM(self, spin_orbs = True):
        """
        Return the 1 electron reduced density matrix P, where
        [P]_pq = <sd|a_p*a_q|sd>
        spin_orbs indicates if the 1RDM is in spin or spatial orbitals
        """
        if spin_orbs:
            density = np.ndarray((self.n_qubit,self.n_qubit))
            for p, q in product(range(self.n_qubit), repeat = 2):
                density[p,q] = self @ (self.Epq(p, q))
        else:
            n = self.n_qubit // 2
            density = np.ndarray((n,n))
            for p, q in product(range(n), repeat = 2):
                density[p,q] = self @ (self.Epq(2 * p, 2 * q) + self.Epq(2*p + 1, 2*q + 1))
        return density

    def variance(self, op: of.FermionOperator):
        """
        Return the variance of the operator on the current state.
        Var_O(sd) = <sd|O^2|sd> - <sd|O|sd>^2
        """
        if isinstance(op, of.FermionOperator):
            return self @ self.Hf_state(op).Hf_state(op) - self.exp(op) ** 2
        elif isinstance(op, np.ndarray):
            return self @ self.tensor_state(op).tensor_state(op) - self.exp(op) ** 2
        else:
            print("Invalid input type")
            return -1

    def get_var_num(self, U = None):
        """
        Return the variance of number operators after acting Mean Field Unitaries on the state:
        unitary_var_num = \sum_i(var_{n_i}(U|sd>)) = \sum_i(var_{Un_iU*}(sd)) if U is defined,
        otherwise U = Identity
        """
        variance = 0
        n = self.n_qubit
        for p in range(n):
            n_op = np.zeros((n, n))
            n_op[p,p] = 1
            # Acting the unitary on the number operator
            if U is not None:
                n_op = np.einsum('ak,bl,kl->ab', np.conj(U), U, n_op)
            variance += self.variance(n_op)
        return variance

    def get_entropy_linear(self, U = None):
        """
        Return the linear entropy of number operators after acting Mean Field Unitaries on the state:
        S = - \sum_k{n_k * log_2(n_k)}, 
        n_k = U @ n_k @ U*, U is set to identity by default.
        """
        S = 0
        for p in range(self.n_qubit):
            n_op = np.zeros((self.n_qubit, self.n_qubit))
            n_op[p,p] = 1
            if U is not None:
                # Acting the unitary on the number operator
                n_op = np.einsum('ak,bl,kl->ab', np.conj(U), U, n_op)
            exp_n = self.exp(n_op)
            if abs(exp_n) > self.eps:
                S -= exp_n * math.log2(exp_n)
        return S

    def get_entropy(self):
        """
        Return the Shannon entropy of the current state
        S = - \sum_k{c_k * log_2(c_k)}, 
        For the current state as |Psi> = \sum_k{c_k|SD_k>}
        """
        S = 0
        for state, coeff in self.dic.items():
            ck = np.real(np.conj(coeff) * coeff)
            if ck > 0:
                S -= ck * math.log2(ck)
        return np.real(S)

    def get_creation_operators(self):
        """Return the creation operator which creates the current state.
        """
        op = of.FermionOperator.zero()
        for s, coeff in self.dic.items():
            # Insert a_p* for p in decreasing order to avoid sign changes
            bits = find_1_bits(s)[::-1]
            tup = tuple((i, 1) for i in bits)
            op += of.FermionOperator(tup, coeff)
        return op

    def CR(self, idx, coeff = 1):
        """Acting creation operator a_p* on the current state, return the resulting state"""
        if self.dic == {}:
            n_qubit = max(self.n_qubit, idx)
            dic = {}
            dic[1 << idx] = coeff
            return sdstate(n_qubit = n_qubit, dic = dic)
        else:
            tmp = {}
            n_qubit = max(self.n_qubit, idx)
            for state, coeff in self.dic.items():
                if not state & (1 << idx):
                    # Create electron in orbital idx
                    cr_state = state ^ (1 << idx)
                    # Count number of electrons before index idx
                    p = parity(state >> (idx + 1))
                    tmp[cr_state] = coeff * ((-1) ** p)
            return sdstate(n_qubit = n_qubit, dic = tmp)

    def MF_unitary(self, U):
        """Acting a Mean Field Unitary on the current state
        """
        # Getting the creation operators which creates the current state
        op = self.get_creation_operators()
        # Acting unitary on the operators
        U_op = of.normal_ordered(transform_CR(op, U))
        # Generated the state from Unitarily transformed operators
        return create_sdstate(U_op, n_qubit = self.n_qubit)

    def get_covariance(self, op1, op2):
        """Return the covariance of operators op1 and op2 on the current state
        Cov(op1, op2) = <SD|op1op2|SD> - <SD|op1|SD><SD|op2|SD>
        """
        if isinstance(op2, np.ndarray):
            tmp = self.tensor_state(op2)
        elif isinstance(op2, of.FermionOperator):
            tmp = self.Hf_state(op2)
        else:
            print("Invalid input type of op2")
            return -1
        if isinstance(op1, np.ndarray):
            tmp = tmp.tensor_state(op1)
        elif isinstance(op1, of.FermionOperator):
            tmp = tmp.Hf_state(op1)
        else:
            print("Invalid input type for op1")
            return -1
        exp12 = self @ tmp
        exp1 = self.exp(op1)
        exp2 = self.exp(op2)
        return exp12 - exp1*exp2

    def get_cov_cost(self, U = None):
        """Return the sum of covariance of number operators on the current state
        """
        cov = 0
        for p in range(self.n_qubit):
            n_p = of.FermionOperator(f"{p}^ {p}")
            if U is not None:
                n_p_vec = np.zeros((self.n_qubit,self.n_qubit))
                n_p_vec[p,p] = 1
                n_p_vec = np.einsum('ak,bl,kl->ab', np.conj(U), U, n_p_vec)
                n_p = n_p_vec
            # Making use of cov(A,B) = cov(B,A), adding cov(p,q) twice to account for cov(q,p)
            for q in range(p + 1):
                if p == q:
                    coeff = 1
                else:
                    coeff = 2
                n_q = of.FermionOperator(f"{q}^ {q}")
                if U is not None:
                    n_q_vec = np.zeros((self.n_qubit,self.n_qubit))
                    n_q_vec[q,q] = 1
                    n_q_vec = np.einsum('ak,bl,kl->ab', np.conj(U), U, n_q_vec)
                    n_q = n_q_vec
                cov += coeff * np.linalg.norm(self.get_covariance(n_p, n_q))
        return cov

    def get_Sz(self):
        """Return the expectation value of Sz operator"""
        Sz = of.hamiltonians.sz_operator(self.n_qubit//2)
        return self.exp(Sz)

    def get_S2(self):
        """Return the expectation value of S2 operator"""
        S2 = of.hamiltonians.s_squared_operator(self.n_qubit//2)
        return self.exp(S2)

    def get_SD_num_p(self, p: float):
        """
        Return the minimum number of Slater Determinants which would sum to a
        total probability of p. Assuming the current state is normalized.
        """
        assert 0 < p <= 1, "Invalid input of probability"
        self.normalize()
        num = 0
        lis = [np.real(i[1] * np.conj(i[1])) for i in self.dic.items()]
        lis.sort(reverse = True)
        tmp = 0
        while tmp < p:
            tmp += lis[num]
            num += 1
        return num

    def get_prob_SD(self, n: int):
        """
        Return the sum of probability for the leading n Slater Determinants. Assuming
        the current state is normalized.
        """
        self.normalize()
        lis = [np.real(i[1] * np.conj(i[1])) for i in self.dic.items()]
        lis.sort(reverse = True)
        return sum(lis[:n])

def transform_CR(op: of.FermionOperator, U: np.ndarray):
    """Transform a series of creation operators with the given MF unitary rotation U
    U @ CR
    CR = \PI_p{a_p*}
    Given the creation operator op, transform it with the Unitary Rotation U.
    """
    n = U.shape[0]
    op_U = of.FermionOperator.zero()
    for tup, val in op.terms.items():
        cur_op = of.FermionOperator.identity()
        lis = [i[0] for i in tup]
        assert not 0 in [i[1] for i in tup], "CR consists of annihilation operator: " + str(of.FermionOperator(tup, val))
        # lis stores the indexes of creation operators
        for i in lis:
            tmp = np.zeros(n)
            # Transform the creation operator into vector representation
            tmp[i] = 1
            # Transform the creation operator with unitary transformation
            u_tmp = np.einsum('q,qp->p', tmp, U)
            # Construct the transformed operator for the current index
            tmp_op = vec_to_CR(u_tmp)
            # Multiply the transformed operator to the current string operator to add cur_op
            cur_op *= tmp_op
        # Update the current operator, together with the coefficient of the operator
        op_U += cur_op * val
    return op_U
   
def vec_to_CR(vec) -> of.FermionOperator:
    """Given a vectorized representation of creation operator, return the linear combination
    of creation operator
    """
    op = of.FermionOperator.zero()
    for i in range(len(vec)):
        if vec[i] != 0:
            op += of.FermionOperator(f"{i}^", vec[i])
    return op
            
def create_sdstate(CR: of.FermionOperator, n_qubit = None):
    """Return the state created from a linear combination of creation operators strings.
    |SD> = \sum_iP{CR_i}|vacuum>
    where each CR_i is a string of creation operators
    CR_i = \PI_p{a_p*}
    """
    if not n_qubit:
        n_qubit = of.count_qubits(CR)
    sd = sdstate(n_qubit = n_qubit)
    for tup, val in CR.terms.items():
        if tup == ():
            sd.dic[0] = val
            continue
        tmp = sdstate(n_qubit = n_qubit)
        lis_idx = [i[0] for i in tup]
        lis_op = [i[1] for i in tup]
        assert not 0 in lis_op, "CR consists of annihilation operator: " + str(of.FermionOperator(tup, val))
        # Act the operators on vacuum in reverse order
        for idx in lis_idx[::-1]:
            tmp = tmp.CR(idx, 1)
        sd += val * tmp
    return sd

def find_1_bits(n: int) -> int:
    """Return the list of indices of 1 in the binary representation of n, in increasing order
    """
    index = 0
    indices = []
    while n > 0:
        # Check if the lowest bit is 1
        if n & 1:
            indices.append(index)
        # Right shift n by 1 to check the next bit
        n >>= 1
        index += 1
    return indices

def int_to_str(k, n_qubit):
    """
    Convert the binary integar to occupied and virtual orbitals, in reversed order and fill
    in the extra virtual orbitals
    >>> int_to_str(1, 2)
    '10'
    >>> int_to_str(2, 3)
    '010'
    >>> int_to_str(3, 3)
    '110'    
    """
    return str(bin(k))[2:][::-1] + "0" * (n_qubit - k.bit_length() - (k == 0)) 

def reverse_bits(number: int, num_bits: int) -> int:
    """Reverse the binary bits of number given total number of bits is num_bits.
    """
    reversed_num = 0
    for i in range(num_bits):
        # Shift reversed_num to the left to make room for the next bit
        reversed_num <<= 1
        # Add the rightmost bit of the number to reversed_num
        reversed_num |= (number >> i) & 1
    return reversed_num


def parity_pq(number: int, a: int, b: int):
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
    result = (number & mask) >> (p + 1)
    return parity(result)

def parity(num: int):
    """Return the parity of bits in a given binary number"""
    parity = 0
    while num:
        parity ^= 1
        num &= num - 1  # Drops the lowest set bit
    return parity

def actable_pq(n: int, p: int, q: int):
    """
    Determines if a_p^a_q annihilates the current state given by n, for n as an index in fock space
    """
    return (p == q and (n & 1 << q) != 0) or ((n & 1 << p) == 0 and (n & 1 << q) != 0)
    