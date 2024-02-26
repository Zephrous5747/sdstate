from module_sdstate.sdstate import *
from modulesdstate.lanczos_utils import *

if __name__ == "__main__":
    n = 3
    sd = sdstate(1, n_qubit = 4)
    sd += sdstate(2, n_qubit = 4)
    assert sd.norm() == 2 ** 0.5