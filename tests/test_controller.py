# external imports
import numpy as np
import pytest

# local imports
from controllers import PIDController


class SimpleSim:
    def __init__(self, K_p=1.0, K_i=0.0, K_d=0.0, dt_anim=0.01):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.dt_anim = dt_anim


def test_calc_u_correct():
    sim = SimpleSim(K_p=2.0, K_i=0.0, K_d=0.0)
    controller = PIDController(sim=sim)
    # error e = y_sp - y_meas
    e = np.array([[1.0 - 0.4]])
    u = controller.calc_u(e)
    assert isinstance(u, np.ndarray)
    assert u.shape == (1, 1)
    assert pytest.approx(u[0, 0], rel=1e-9) == pytest.approx(1.2, rel=1e-9)


def test_calc_u_shape_error():
    sim = SimpleSim(K_p=1.0)
    controller = PIDController(sim=sim)
    # wrong shapes should raise ValueError
    with pytest.raises(ValueError):
        controller.calc_u(np.array([1.0]))

    with pytest.raises(ValueError):
        controller.calc_u(np.array([[1.0, 2.0]]))
