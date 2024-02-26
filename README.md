# sdstate

Implementation of Slater Determinant states as dictionaries of states and coefficient pairs. Each state is represented with an integar treated as binary, allowing applying of Excitation operators on the state efficiently. Memory-efficient implementation of Lanczos iteration for estimating Hamiltonian spectrum range.

## Installation

```bash
$ pip install module_sdstate
```

## Usage

- Memory efficient implementation of Slater-Derterminant states.
- Compute expectation value of Hamiltonian on Slater-Determinant states states.
- Compute Hartree-Fock energy estimation of a Hamiltonian.
- Efficient estimation of ground state energy and spectrum range of Hamiltonian.
- Compactable with Hamiltonian represented by openfermion.FermionOperator and a tuple of 1-electron and 2-electron tensor.

## License

`sdstate` was created by Linjun Wang <linjun.wang@mail.utoronto.ca>. It is licensed under the terms of the MIT license.

## Credits

`sdstate` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
