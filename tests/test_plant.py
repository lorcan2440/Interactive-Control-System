import numpy as np


def test_simulation_initialises_correctly(sim):
    '''
    Example of another test that reuses the same fixture.
    '''
    assert sim is not None
    assert sim.plant is not None


def test_plant_transfer_functions_equivalent(sim):
    '''
    Verify that the two methods of computing the transfer function of the plant are equivalent.
    The function computes G(jw) over a range of frequencies w,
    using the state space matrices and the hardcoded TF.
    '''
    A, B2, C, _ = sim.plant.plant_ss_matrices()

    plant_tf_from_ss = lambda s: (C @ np.linalg.inv(s * np.eye(A.shape[0]) - A) @ B2)[0][0]

    freq_range = np.logspace(-2, 6, 20, base=10)
    g1 = [plant_tf_from_ss(complex(0, 1) * w) for w in freq_range]
    g2 = [sim.plant.plant_tf(complex(0, 1) * w) for w in freq_range]

    assert np.allclose(g1, g2)
