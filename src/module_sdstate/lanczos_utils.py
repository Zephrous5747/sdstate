"""
Implements Hartree-Fock and Lanczos iterations to approximate energy range 
of an electronic Hamiltonian.
"""
import numpy as np
import copy
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool
import openfermion as of
import os

# Assuming your modules are structured like this
from module_sdstate.sdstate_utils import sdstate
from module_sdstate.pauli_utils import apply_qubit_operator

def lanczos(Hf, steps, state=None, ne=None):
    """
    Applies Lanczos iteration on the given Hamiltonian.
    
    Supports:
        - openfermion.FermionOperator
        - openfermion.QubitOperator
        - Tuple of (1e, 2e) tensors
        
    Args:
        Hf: The Hamiltonian.
        steps: Number of Lanczos iterations.
        state: Initial sdstate vector (optional).
        ne: Number of electrons for default initialization (optional).
        
    Returns:
        states: Normalized state vectors from each iteration.
        A: Main diagonal elements of the tridiagonal matrix.
        B: Sub-diagonal elements of the tridiagonal matrix.
    """
    # 1. Determine Hamiltonian Type and Qubit Count
    is_tuple = False
    is_qubit = False
    is_fermion = False
    
    if isinstance(Hf, tuple):
        assert len(Hf) == 2, "Incorrect input of Hf tuple."
        is_tuple = True
        n_qubits = Hf[0].shape[0]
    elif isinstance(Hf, of.QubitOperator):
        is_qubit = True
        n_qubits = of.utils.count_qubits(Hf)
    elif isinstance(Hf, of.FermionOperator):
        is_fermion = True
        n_qubits = of.utils.count_qubits(Hf)
    else:
        raise TypeError("Unsupported Hamiltonian type.")
        
    # 2. Setup Initial State if not provided
    if state is None:
        if ne is None:
            ne = n_qubits // 2
        # Default initialization: Superposition of two extreme occupations
        state_int_1 = int("1" * ne + "0" * (n_qubits - ne), 2)
        state_int_2 = int("0" * (n_qubits - ne) + "1" * ne, 2)
        state = sdstate(state_int_1, n_qubit=n_qubits)
        state += sdstate(state_int_2, n_qubit=n_qubits)
        state.normalize()
        
    # 3. First Iteration (Initialization)
    # Apply Hamiltonian based on type
    if is_tuple:
        tmp = state.tensor_state(Hf[0]) + state.tensor_state(Hf[1])
    elif is_qubit:
        tmp = apply_qubit_operator(state, Hf)
    elif is_fermion:
        tmp = state.Hf_state(Hf)
        
    # Use real here to get only real part from complex, which is physical
    # ai are the diagonal elements, corresponding to expectation values of H on the state
    # bi are the norms, which also have to real
    ai = np.real(tmp @ state)
    tmp -= ai * state
    
    A = [ai]
    B = []
    states = [state]
    vi = tmp
    
    # 4. Main Lanczos Loop
    for i in range(1, steps):
        bi = tmp.norm()
        
        # Handle early convergence or exact invariant subspace
        if bi < state.eps:
            break
            
        vi = tmp / bi
        
        # Apply Hamiltonian based on type
        if is_tuple:
            tmp = vi.tensor_state(Hf[0]) + vi.tensor_state(Hf[1])
        elif is_qubit:
            tmp = apply_qubit_operator(vi, Hf)
        elif is_fermion:
            tmp = vi.Hf_state(Hf)
            
        ai = np.real(vi @ tmp)
        
        # Orthogonalize against current and previous vectors
        tmp -= ai * vi 
        tmp -= bi * states[i - 1]
        
        states.append(vi)
        A.append(ai)
        B.append(bi)
        
    return states, A, B


def lanczos_range(Hf, steps, state=None, ne=None):
    """
    Returns the largest and smallest eigenvalue from Lanczos iterations.
    Strongly recommend inputting the number of electrons (ne) for the 
    ground state to find the correct spectral range.
    """
    _, A, B = lanczos(Hf, steps=steps, state=state, ne=ne)
    
    # Check if the tridiagonal matrix is large enough to diagonalize
    if len(A) == 0:
        return 0.0, 0.0
    elif len(B) == 0:
        return A[0], A[0]
        
    eigs, _ = eigh_tridiagonal(np.real(A), np.real(B))
    return max(eigs), min(eigs)


def lanczos_total_range(Hf, steps=2, states=[], e_nums=[], multiprocessing=True):
    """
    Returns the largest and smallest eigenvalue from Lanczos iterations 
    across multiple electron number subspaces.
    """
    # Determine n_qubits safely for all types
    if isinstance(Hf, tuple):
        n = Hf[0].shape[0]
    else:
        n = of.utils.count_qubits(Hf)
        
    if multiprocessing:
        num_processes = os.cpu_count()
        with Pool(processes=num_processes) as pool:
            if len(states) != 0:
                res = pool.starmap(lanczos_range, [(Hf, steps, st, None) for st in states])
            elif len(e_nums) != 0:
                res = pool.starmap(lanczos_range, [(Hf, steps, None, ne) for ne in e_nums])
            else:
                res = pool.starmap(lanczos_range, [(Hf, steps, None, ne) for ne in range(n + 1)])
                
        if not res:
            return 0.0, 0.0
            
        E_max = max([i[0] for i in res])
        E_min = min([i[1] for i in res])
    else:
        E_max = -1e10
        E_min = 1e10
        if len(states) != 0:
            for st in states:
                e_max_local, e_min_local = lanczos_range(Hf, steps=steps, state=st)
                E_max = max(E_max, e_max_local)
                E_min = min(E_min, e_min_local)
        else:
            search_space = e_nums if len(e_nums) != 0 else range(n + 1)
            for ne in search_space:
                e_max_local, e_min_local = lanczos_range(Hf, steps=steps, ne=ne)
                E_max = max(E_max, e_max_local)
                E_min = min(E_min, e_min_local)
                
    return E_max, E_min