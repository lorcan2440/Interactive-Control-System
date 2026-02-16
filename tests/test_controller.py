# external imports
import numpy as np
import pytest

# local imports
from controllers import (
    ManualController,
    OpenLoopController,
    BangBangController,
    PIDController,
)

# TODO: improve these tests by testing real numbers manually


class SimpleSim:
    def __init__(
        self,
        K_p=1.0,
        K_i=0.0,
        K_d=0.0,
        dt_anim=0.01,
        manual_u=0.0,
        y_sp=0.0,
        U_plus=1.0,
        U_minus=-1.0,
    ):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.dt_anim = dt_anim
        self.manual_u = manual_u
        self.y_sp = y_sp
        self.U_plus = U_plus
        self.U_minus = U_minus


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


def test_manual_controller_returns_manual_u():
    sim = SimpleSim(manual_u=3.14)
    controller = ManualController(sim=sim)
    u = controller.calc_u()
    assert isinstance(u, np.ndarray)
    assert u.shape == (1, 1)
    assert pytest.approx(u[0, 0], rel=1e-9) == pytest.approx(3.14, rel=1e-9)


def test_openloop_controller_computes_feedforward():
    # create a trivial 1x1 plant with steady-state gain = C * inv(-A) * B
    class Plant:
        def __init__(self):
            self.dims = 1
            self.A = np.array([[-2.0]])
            self.B = np.array([[1.0]])
            self.C = np.array([[4.0]])
            self.D = np.array([[0.0]])

    sim = SimpleSim(y_sp=5.0)
    plant = Plant()
    controller = OpenLoopController(sim=sim, plant=plant)
    u = controller.calc_u()
    assert isinstance(u, np.ndarray)
    assert u.shape == (1, 1)
    # steady-state gain = 4 * (1/2) = 2 => u = 5 / 2 = 2.5
    assert pytest.approx(u[0, 0], rel=1e-9) == pytest.approx(2.5, rel=1e-9)


def test_bangbang_controller_on_off_behaviour():
    sim = SimpleSim(U_plus=10.0, U_minus=-7.0)
    controller = BangBangController(sim=sim)

    u_pos = controller.calc_u(0.2)
    assert isinstance(u_pos, np.ndarray)
    assert u_pos.shape == (1, 1)
    assert pytest.approx(u_pos[0, 0], rel=1e-9) == pytest.approx(10.0, rel=1e-9)

    u_neg = controller.calc_u(-0.5)
    assert isinstance(u_neg, np.ndarray)
    assert u_neg.shape == (1, 1)
    assert pytest.approx(u_neg[0, 0], rel=1e-9) == pytest.approx(-7.0, rel=1e-9)

    u_zero = controller.calc_u(0.0)
    assert isinstance(u_zero, np.ndarray)
    assert u_zero.shape == (1, 1)
    assert pytest.approx(u_zero[0, 0], rel=1e-9) == pytest.approx(0.0, rel=1e-9)


def test_pid_controller_p_i_d_terms_and_memory():
    sim = SimpleSim(K_p=2.0, K_i=1.0, K_d=0.5, dt_anim=0.1)
    controller = PIDController(sim=sim)

    e = np.array([[1.0]])
    u1 = controller.calc_u(e)
    # u_p = 2 * 1 = 2
    # u_i = 1 * (1 * 0.1) = 0.1
    # u_d = 0.5 * ((1 - 0) / 0.1) = 0.5 * 10 = 5
    assert pytest.approx(u1[0, 0], rel=1e-9) == pytest.approx(3.1, rel=1e-9)

    # second call with same error: derivative term should be zero, integral doubles
    u2 = controller.calc_u(e)
    # u_p = 2
    # u_i = 1 * (2 * 0.1) = 0.2
    # u_d = 0
    assert pytest.approx(u2[0, 0], rel=1e-9) == pytest.approx(3.0, rel=1e-9)
