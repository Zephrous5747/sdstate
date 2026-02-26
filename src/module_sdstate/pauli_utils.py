"""
pauli_utils.py

Provides utility functions to apply OpenFermion QubitOperators directly onto 
the custom sdstate dictionary representation without constructing dense or 
sparse matrices. This supports matrix-free Lanczos iterations for large Hilbert spaces.
"""
import openfermion as of
from module_sdstate.sdstate_utils import sdstate

def apply_pauli_string(state, pauli_string: tuple, term_coeff: complex):
    """
    Applies a single Pauli string (e.g., 'X0 Y1 Z2') to the given sdstate.
    
    This function mimics the logic of `.Epqrs` in the sdstate class but is 
    specifically designed for the computational basis representation of qubits.
    It directly manipulates the bitstring integers representing the occupied states.
    
    Args:
        state: An instance of the sdstate class.
        pauli_string: A tuple representing the Pauli operator in OpenFermion format.
                      Format: ((qubit_index_1, 'P_1'), (qubit_index_2, 'P_2'), ...)
                      Example: ((0, 'X'), (2, 'Z'))
        term_coeff: The complex or real coefficient of the Pauli string.
        
    Returns:
        A new sdstate instance representing the resulting state vector.
    """    
    # Initialize a new state to store the result of the operator application
    out_state = sdstate(n_qubit=state.n_qubit)
    
    # Iterate over all non-zero computational basis states in the current wavefunction
    for state_int, state_coeff in state.dic.items():
        new_int = state_int
        new_coeff = state_coeff * term_coeff
        
        # Apply the Pauli operator site by site onto the current basis state
        for qubit_idx, pauli_char in pauli_string:
            
            # Pauli X: Bit flip (0 -> 1, 1 -> 0). 
            # Implemented via bitwise XOR.
            if pauli_char == 'X':
                new_int ^= (1 << qubit_idx)
                
            # Pauli Z: Phase flip.
            # If the bit is 1, multiply the coefficient by -1. If 0, do nothing.
            # Implemented via bitwise AND to check occupancy.
            elif pauli_char == 'Z':
                if new_int & (1 << qubit_idx):
                    new_coeff *= -1
                    
            # Pauli Y: Combination of bit flip and phase flip with imaginary phase.
            # Y|0> = i|1>, Y|1> = -i|0>
            elif pauli_char == 'Y':
                if new_int & (1 << qubit_idx):
                    # State is |1>. Flip to |0> and multiply phase by -i.
                    new_coeff *= -1j
                else:
                    # State is |0>. Flip to |1> and multiply phase by +i.
                    new_coeff *= 1j
                
                # Execute the bit flip via XOR
                new_int ^= (1 << qubit_idx)
                
        # Accumulate the resulting determinant into the new state's dictionary.
        # Direct dictionary access is used here to avoid the overhead of 
        # repeatedly instantiating and adding sdstate objects within the loop.
        if new_int in out_state.dic:
            out_state.dic[new_int] += new_coeff
        else:
            out_state.dic[new_int] = new_coeff
            
    # Clean up any computational zero states resulting from phase cancellations
    out_state.remove_zeros()
    return out_state


def apply_qubit_operator(state, qubit_op: of.QubitOperator):
    """
    Applies a full OpenFermion QubitOperator (e.g., a mapped Hamiltonian) 
    to the given sdstate.
    
    This function iterates through all terms in the QubitOperator and 
    accumulates the results, mimicking the `.Hf_state` method for FermionOperators.
    
    Args:
        state: An instance of the sdstate class.
        qubit_op: An openfermion.QubitOperator representing the observable.
        
    Returns:
        A new sdstate instance representing the resulting state vector.
    """
        
    # Initialize the total resulting state vector
    result_state = sdstate(n_qubit=state.n_qubit)
    
    # Iterate through each Pauli string and its corresponding coefficient in the Hamiltonian
    for pauli_string, coeff in qubit_op.terms.items():
        # Handle the Identity operator (empty tuple in OpenFermion)
        if not pauli_string:
            result_state += state * coeff
        else:
            # Apply the non-trivial Pauli string and accumulate
            result_state += apply_pauli_string(state, pauli_string, coeff)
            
    return result_state