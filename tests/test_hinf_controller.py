import numpy as np
import pytest

from controllers import HInfinityController
from integrators import IntegratorType
from plant import Plant
from utils import EPS, PLANT_DEFAULT_PARAMS, TIME_STEPS, GUI_SLIDER_CONFIG


class SimpleSim:
    def __init__(self):
        self.dt_anim = 0.01
        self.integrator_method = IntegratorType.ANALYTIC_SDE
        self.use_ode_mode = False
        self.y_sp = np.array([[0.5]])
        self.Hinf_C1_x1 = 1.0
        self.Hinf_C1_u = 1.0


def make_controller():
    plant = Plant(
        dims=1,
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.1]]),
        R=np.array([[0.1]]),
    )
    return HInfinityController(SimpleSim(), plant)


def make_default_plant_controller():
    simulator = SimpleSim()
    simulator.dt_anim = TIME_STEPS['DT_ANIM']
    simulator.Hinf_C1_x1 = 2.0
    simulator.Hinf_C1_x2 = 2.0
    return HInfinityController(simulator, Plant(**PLANT_DEFAULT_PARAMS))


def test_hinf_design_and_control_output():
    controller = make_controller()

    controller.update_controller_matrices()
    control = controller.calc_u(np.array([[0.5]]))

    assert np.isfinite(controller.gamma)
    assert controller.gamma > 0.0
    assert control.shape == (1, 1)
    assert np.all(np.isfinite(control))
    assert controller.A_Kd.shape == (1, 1)
    assert controller.B_Kd.shape == (1, 1)
    assert controller.is_closed_loop_stable_discrete()[0]


def test_default_plant_is_stabilizable_detectable_and_care_design_succeeds():
    controller = make_default_plant_controller()

    assert np.allclose(
        controller.plant.Q,
        GUI_SLIDER_CONFIG['w_process_stddev']['init'] ** 2 * np.eye(controller.plant.dims),
    )
    assert np.allclose(controller.plant.R, [[GUI_SLIDER_CONFIG['w_meas_stddev']['init'] ** 2]])
    assert controller.check_stabilisability_and_detectability() == (True, True)
    controller.update_controller_matrices()

    assert np.isfinite(controller.gamma)
    assert controller.gamma > 0.0
    assert controller.is_closed_loop_stable_discrete()[0]

    equilibrium_matrix = np.block([
        [controller.plant.A, controller.plant.B],
        [controller.plant.C, controller.plant.D],
    ])
    equilibrium_rhs = np.vstack([
        np.zeros((controller.plant.dims, 1)),
        controller.simulator.y_sp,
    ])
    open_loop_u = np.linalg.solve(equilibrium_matrix, equilibrium_rhs)[-1:]
    first_u = controller.calc_u(np.array([[0.5]]))
    assert not np.allclose(first_u, open_loop_u)


def test_hinf_input_weight_changes_controller_with_default_noise():
    low_penalty = make_default_plant_controller()
    high_penalty = make_default_plant_controller()
    low_penalty.simulator.Hinf_C1_u = 0.1
    high_penalty.simulator.Hinf_C1_u = 10.0

    low_penalty.update_controller_matrices()
    high_penalty.update_controller_matrices()

    assert not np.allclose(low_penalty.F, high_penalty.F)
    assert not np.allclose(low_penalty.B_Kd, high_penalty.B_Kd)


@pytest.mark.parametrize(
    ('B', 'C', 'expected'),
    [
        (np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]]), (False, True)),
        (np.array([[1.0], [0.0]]), np.array([[0.0, 1.0]]), (True, False)),
    ],
)
def test_hinf_plant_check_rejects_unstable_uncontrollable_or_unobservable_modes(B, C, expected):
    simulator = SimpleSim()
    plant = Plant(
        dims=2,
        A=np.diag([1.0, -2.0]),
        B=B,
        C=C,
        D=np.array([[0.0]]),
        Q=np.zeros((2, 2)),
        R=np.array([[0.1]]),
    )
    controller = HInfinityController(simulator, plant)

    assert controller.check_stabilisability_and_detectability() == expected
