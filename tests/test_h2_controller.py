import numpy as np
from unittest.mock import patch

from controllers import H2Controller
from integrators import IntegratorType
from plant import Plant


class SimpleSim:
    def __init__(self, dims, y_sp=1.0, dt_anim=0.01):
        self.dt_anim = dt_anim
        self.integrator_method = IntegratorType.EULER_MARUYAMA
        self.use_ode_mode = False
        self.y_sp = np.array([[y_sp]], dtype=float)
        self.C1_u = 1.0
        for i in range(dims):
            setattr(self, f'C1_x{i + 1}', 1.0)


def test_h2_controller_returns_array_and_applies_negative_state_feedback():
    plant = Plant(
        dims=1,
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[0.0]]),
    )
    sim = SimpleSim(dims=1)
    controller = H2Controller(simulator=sim, plant=plant)

    u_at_zero_state = controller.calc_u(np.array([[1.0]]))
    assert u_at_zero_state.shape == (1, 1)
    assert u_at_zero_state[0, 0] > 1.0

    controller.x_hat = np.array([[2.0]])
    u_above_target = controller.calc_u(np.array([[-1.0]]))
    assert u_above_target[0, 0] < 1.0


def test_h2_controller_uses_all_plant_states_and_rebuilds_changed_weights():
    A = np.array([
        [-1.0, 1.0, 0.0],
        [0.0, -2.0, 1.0],
        [0.0, 0.0, -3.0],
    ])
    plant = Plant(
        dims=3,
        A=A,
        B=np.array([[0.0], [0.0], [1.0]]),
        C=np.array([[1.0, 0.0, 0.0]]),
        D=np.array([[0.0]]),
        Q=np.zeros((3, 3)),
        R=np.array([[0.0]]),
    )
    sim = SimpleSim(dims=3)
    controller = H2Controller(simulator=sim, plant=plant)

    first_u = controller.calc_u(np.array([[1.0]]))
    first_gain = controller.F.copy()
    assert first_u.shape == (1, 1)
    assert first_gain.shape == (1, 3)

    sim.C1_x2 = 5.0
    controller.calc_u(np.array([[1.0]]))
    assert not np.allclose(controller.F, first_gain)

    state_weighted_gain = controller.F.copy()
    sim.C1_u = 2.0
    controller.calc_u(np.array([[1.0]]))
    assert not np.allclose(controller.F, state_weighted_gain)


def test_h2_closed_loop_stability_returns_discrete_poles():
    plant = Plant(
        dims=1,
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[0.0]]),
    )
    controller = H2Controller(simulator=SimpleSim(dims=1), plant=plant)
    controller.update_controller_matrices()

    stable, poles = controller.is_closed_loop_stable_discrete()

    assert stable
    assert poles.size == 2 * plant.dims + 1
    assert np.all(np.abs(poles) < 1.0)


def test_h2_calc_u_logs_transition_to_instability():
    plant = Plant(
        dims=1,
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[0.0]]),
    )
    controller = H2Controller(simulator=SimpleSim(dims=1), plant=plant)
    controller.update_controller_matrices()
    controller.cl_stable_prev = True

    unstable_result = (False, np.array([1.1 + 0.0j]))
    with patch.object(controller, 'is_closed_loop_stable_discrete', return_value=unstable_result), \
            patch.object(controller.logger, 'warning') as warning:
        controller.calc_u(np.array([[1.0]]))
        controller.calc_u(np.array([[1.0]]))

    warning.assert_called_once()
