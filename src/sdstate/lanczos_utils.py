"""Implements Hartree-Fock and Lanczos iterations to approximate energy range of an electronic Hamiltonian"""
import numpy as np
import copy
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool
import openfermion as of
import os
from sdstate import *

def HF_energy(Hf, n, ne):
    """Find the energy of largest and smallest slater determinant states with Hf as Fermionic Hamiltonian and
     number of electrons as ne.
    """    
    #  <low|H|low>
    lstate = sdstate(((1 << ne) - 1) << (n-ne), n_qubit = n)
    E_low = lstate.exp(Hf)
    #  <high|H|high>
    hstate = sdstate((1 << ne) - 1, n_qubit = n)
    E_high = hstate.exp(Hf)
    return E_high, E_low

def HF_spectrum_range(Hf, multiprocessing = True):
    """Compute the naive Hartree-Fock energy range of the Hamiltonian 2e tensor Hf for all number of electrons.
    Multiprocessing parameter is set to parallelize computations for the states with different number of electrons. 
    Warning: This is not the actual Hartree-Fock energy range, which takes exponential time to compute
    """
    n = of.utils.count_qubits(Hf)
    if multiprocessing:
        num_processes = os.cpu_count()
        with Pool(processes=num_processes) as pool:
            res = pool.starmap(HF_energy, [(Hf, n, ne) for ne in range(n)])
        low = 1e10
        low_state = ""
        high = -1e10
        high_state = ""
        for ne in range(len(res)):
            E_high = res[ne][0]
            E_low = res[ne][1]
            if E_low < low:
                low_state = (1 << ne) - 1
                low = E_low
            if E_high > high:
                high_state = ((1 << ne) - 1) << (n-ne)
                high = E_high
    else:
        low = 1e10
        low_state = ""
        high = -1e10
        high_state = ""
        for ne in range(n):
            low_int = (1 << ne) - 1
            lstate = sdstate(low_int, n_qubit = n)
    #         <low|H|low>
            E_low = lstate.exp(Hf)
            high_int = ((1 << ne) - 1) << (n-ne)
            hstate = sdstate(high_int, n_qubit = n)
    #         <high|H|high>
            E_high = hstate.exp(Hf)
            if E_low < low:
                low_state = low_int
                low = E_low
            if E_high > high:
                high_state = high_int
                high = E_high
    high_str = bin(high_state)[2:][::-1]
    low_str = bin(low_state)[2:][::-1]
    high_str = "0" * (n - len(high_str)) + high_str
    low_str += "0" * (n - len(low_str))
    print("HF E_max: {}".format(high))
    print("HF E_min: {}".format(low))
    return high_str, low_str, high, low


def lanczos(Hf, steps, state = None, ne = None):
    """Applies lanczos iteration on the given FermionOperator Hf, or a tuple of 1e and 2e tensor with number of steps 
    given by steps,
    with initial state as input or number of electrons as input ne.
    Returns normalized states in each iteration, and a tridiagonal matrix with main diagonal in A and sub-diagonal in B.
    """
    flag_tuple = False
    if isinstance(Hf, tuple):
        assert len(Hf) == 2, "Incorrect input of Hf"
        flag_tuple = True
        n_qubits = Hf[0].shape[0]
    else:
        n_qubits = of.utils.count_qubits(Hf)
        
    if state == None:
        if ne == None:
            ne = n_qubits // 2
        state = sdstate(int("1"*ne + "0"*(n_qubits - ne), 2), n_qubit = n_qubits)
        state += sdstate(int("0"*(n_qubits - ne) + "1" * ne, 2), n_qubit = n_qubits)
    state.normalize()
    if flag_tuple:
        tmp = state.tensor_state(Hf[0]) + state.tensor_state(Hf[1]) 
    else:
        tmp = state.Hf_state(Hf)
    ai = tmp @ state
    tmp -= ai * state
    A = [ai]
    B = []
    states = [state]
    vi = tmp
    for i in range(1,steps):
        bi = tmp.norm()
        if bi != 0:
            vi = tmp / bi
        if flag_tuple:
            tmp = vi.tensor_state(Hf[0]) + vi.tensor_state(Hf[1]) 
        else:
            tmp = vi.Hf_state(Hf)
        ai = vi @ tmp
        tmp -= ai * vi 
        tmp -= bi * states[i - 1]
        states.append(vi)
        A.append(ai)
        B.append(bi)
    return states, A, B

def lanczos_range(Hf, steps, state = None, ne = None):
    """ Returns the largest and the smallest eigenvalue from Lanczos iterations with given number of steps,
    number of electrons or initial state. 
    Strongly recommend to input the number of electrons in the ground state to find for correct spetral range
    """
    _, A, B = lanczos(Hf, steps = steps, state = state, ne = ne)
    eigs, _ = eigh_tridiagonal(A,B)
    return max(eigs), min(eigs)

def lanczos_total_range(Hf, steps = 2, states = [], e_nums = [], multiprocessing = True):
    """Returns the largest and the smallest eigenvalue from Lanczos iterations with given number of steps,
    for all possible number of electrons. Multiprocessing will parallelize the computation for all possible 
    number of electrons. states specifies the initial states for the Hamiltonian.
    e_nums is an indicator for the number of electrons subspaces to check for the highest and lowest energy. If 
    not specified states or e_nums, the function will search through all possible values. Steps is set to 2 on default as 
    experimented that 2 iterations is capable of estimating within an accuracy of about 95%.
    """
    if isinstance(Hf, of.FermionOperator):
        n = of.utils.count_qubits(Hf)
    else:
        n = Hf[0].shape[0]
    if multiprocessing:
        num_processes = os.cpu_count()
        with Pool(processes=num_processes) as pool:
            if len(states) != 0:
                res = pool.starmap(lanczos_range, [(Hf, steps, st, None) for st in states])
            elif len(e_nums) != 0:
                res = pool.starmap(lanczos_range, [(Hf, steps, None, ne) for ne in e_nums])
            else:
                res = pool.starmap(lanczos_range, [(Hf, steps, None, ne) for ne in range(n)])
        E_max = max([i[0] for i in res])
        E_min = min([i[1] for i in res])
    else:
        E_max = -1e10
        E_min = 1e10
        if len(states) != 0:
            for st in states:
                states, A, B = lanczos(Hf, steps = steps, state = state)
                eigs, _ = eigh_tridiagonal(A,B)
                E_max = max(max(eigs), E_max)
                E_min = min(min(eigs), E_min)
        else:
            for ne in range(n):
                states, A, B = lanczos(Hf, steps = steps, ne = ne)
                eigs, _ = eigh_tridiagonal(A,B)
                E_max = max(max(eigs), E_max)
                E_min = min(min(eigs), E_min)
    return E_max, E_min