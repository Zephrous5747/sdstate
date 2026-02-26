"""
test_pauli_utils.py

Comprehensive unit tests to verify the matrix-free QubitOperator application.
Ensures perfect agreement with FermionOperator application (using chemist ordering)
across various types of Hamiltonian terms (1-body, 2-body, diagonal, and complex).
"""

import openfermion as of
import numpy as np
# Assuming your original class is in module_sdstate.py
from module_sdstate.sdstate_utils import sdstate
# Assuming the new utility functions are in qubit_utils.py
from module_sdstate.pauli_utils import apply_qubit_operator

def verify_mapping(test_name: str, initial_state: sdstate, fermion_ham: of.FermionOperator):
    """
    Helper function to run a single comparison test between Fermionic 
    and Jordan-Wigner Qubit operations.
    
    Args:
        test_name: String identifier for the test case.
        initial_state: The starting sdstate vector.
        fermion_ham: The raw FermionOperator Hamiltonian.
    """
    print(f"========== Running Test: {test_name} ==========")
    
    # Enforce chemist ordering as required by the sdstate.Hf_state underlying logic
    # This groups creation/annihilation operators to match the Epqrs excitation logic
    fermion_ham_ordered = of.chemist_ordered(fermion_ham)
    
    # 1. Apply Fermionic Hamiltonian Directly (Baseline)
    state_fermionic = initial_state.Hf_state(fermion_ham_ordered)
    
    # 2. Map to Qubit space via Jordan-Wigner and apply
    jw_hamiltonian = of.transforms.jordan_wigner(fermion_ham_ordered)
    state_jw = apply_qubit_operator(initial_state, jw_hamiltonian)
    
    # 3. Verify equivalence by calculating the norm of the difference vector
    diff_vector = state_fermionic - state_jw
    norm_diff = diff_vector.norm()
    
    print(f"Difference Norm: {norm_diff:.2e}")
    if norm_diff < 1e-8:
        print(">>> PASS\n")
    else:
        print(">>> FAIL")
        print("Fermionic Result:")
        print(state_fermionic)
        print("Qubit Result:")
        print(state_jw)
        raise AssertionError(f"Test '{test_name}' failed with norm diff {norm_diff}")


def run_comprehensive_tests():
    """
    Executes a suite of benchmark tests for the matrix-free CI module.
    """
    n_qubits = 6
    
    # Define a superposition initial state to test linearity and phase interference
    # |101000> (int 40) + 0.5j * |010100> (int 20)
    init_state = sdstate(s=40, coeff=1.0, n_qubit=n_qubits)
    init_state += sdstate(s=20, coeff=0.5j, n_qubit=n_qubits)
    init_state.normalize()

    # --- Test Case 1: Pure Diagonal Terms (Number Operators) ---
    # H = 2.0 * n_1 + 1.5 * n_3 + 0.8 * n_5
    # This tests the Z-operator phase logic without altering the basis states.
    ham_diagonal = of.FermionOperator('1^ 1', 2.0)
    ham_diagonal += of.FermionOperator('3^ 3', 1.5)
    ham_diagonal += of.FermionOperator('5^ 5', 0.8)
    verify_mapping("Diagonal Number Operators", init_state, ham_diagonal)

    # --- Test Case 2: 1-Body Hopping Terms (Real and Complex) ---
    # H = (a_1^ a_2 + a_2^ a_1) + 0.5j * (a_3^ a_4 - a_4^ a_3)
    # This rigorously tests X and Y Pauli string applications and parity checks.
    ham_1body = of.FermionOperator('1^ 2', 1.0)
    ham_1body += of.FermionOperator('2^ 1', 1.0)
    ham_1body += of.FermionOperator('3^ 4', 0.5j)
    ham_1body += of.FermionOperator('4^ 3', -0.5j)
    verify_mapping("1-Body Hopping (Real & Complex)", init_state, ham_1body)

    # --- Test Case 3: Standard 2-Body Coulomb-like Terms ---
    # H = (ij|kl) a_i^ a_k^ a_l a_j 
    # This tests the full interaction logic, corresponding to Epqrs.
    ham_2body = of.FermionOperator('0^ 1^ 2 3', 0.5)
    ham_2body += of.FermionOperator('3^ 2^ 1 0', 0.5)  # Hermitian conjugate
    ham_2body += of.FermionOperator('1^ 2^ 4 5', -0.75)
    verify_mapping("2-Body Coulomb Interactions", init_state, ham_2body)

    # --- Test Case 4: Long-Range 2-Body Terms (Testing Parity JW Strings) ---
    # H = a_0^ a_5^ a_4 a_1
    # This tests operators that span across many qubits, generating long Z-strings
    # in the Jordan-Wigner representation, ensuring phase parity is perfectly mapped.
    ham_long_range = of.FermionOperator('0^ 5^ 4 1', 1.25)
    verify_mapping("Long-Range 2-Body Interactions", init_state, ham_long_range)
    
    # --- Test Case 5: Full Random Molecular-like Hamiltonian ---
    # Combining all the above to test accumulation and zero-cancellation
    ham_full = ham_diagonal + ham_1body + ham_2body + ham_long_range
    verify_mapping("Combined Full Hamiltonian", init_state, ham_full)

    print("All tests completed successfully! Qubit matrix-free logic is 100% sound.")

if __name__ == "__main__":
    run_comprehensive_tests()