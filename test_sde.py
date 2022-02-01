import pytest

def test_sde_import():
    from sde import CIMSDE

def test_sde_instantiate_allargs():
    from sde import CIMSDE
    C = CIMSDE(dim=10, batch_size=1, device='cpu')

def test_sde_instantiate_args():
    from sde import CIMSDE
    C = CIMSDE(10)

def test_sde_iteration():
    from sde import CIMSDE
    C = CIMSDE(10)
    C.iteration()

def test_sde_1k_iteration():
    from sde import CIMSDE
    C = CIMSDE(10)
    for i in range(1000):
        C.iteration()