from types import SimpleNamespace
import numpy as np
import pytest

from controllers import ControllerType, PIDController
from integrators import IntegratorType
from plant import Plant
from utils import PLANT_DEFAULT_PARAMS, TIME_STEPS


def _make_pid_controller(tau: float) -> PIDController:
    plant_params = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in PLANT_DEFAULT_PARAMS.items()
    }
    plant = Plant(**plant_params)
    sim = SimpleNamespace(
        K_p=200.0,
        K_i=20.0,
        K_d=50.0,
        tau=tau,
        dt_anim=TIME_STEPS['DT_ANIM'],
        integrator_method=IntegratorType.EULER_MARUYAMA,
        use_ode_mode=False,
        controller_type=ControllerType.PID,
    )
    return PIDController(sim=sim, plant=plant)


def test_discrete_stability_matches_sampled_pid_boundary():
    unstable_controller = _make_pid_controller(tau=0.23)
    stable_controller = _make_pid_controller(tau=0.24)

    unstable, unstable_poles = unstable_controller.is_closed_loop_stable_discrete()
    stable, stable_poles = stable_controller.is_closed_loop_stable_discrete()

    assert not unstable
    assert np.max(np.abs(unstable_poles)) > 1.0
    assert stable
    assert np.max(np.abs(stable_poles)) < 1.0


def test_discrete_stability_uses_current_simulator_gains():
    controller = _make_pid_controller(tau=0.24)

    controller.sim.tau = 0.23
    stable, poles = controller.is_closed_loop_stable_discrete()

    assert not stable
    assert np.max(np.abs(poles)) > 1.0


@pytest.mark.parametrize('filtered_derivative', [True, False])
def test_discrete_stability_poles_are_independent_of_derivative_reference(
    filtered_derivative,
):
    measurement_derivative_controller = _make_pid_controller(tau=0.24)
    error_derivative_controller = _make_pid_controller(tau=0.24)
    measurement_derivative_controller.sim.PID_filtered_derivative = filtered_derivative
    error_derivative_controller.sim.PID_filtered_derivative = filtered_derivative
    measurement_derivative_controller.sim.PID_derivative_on_measurement = True
    error_derivative_controller.sim.PID_derivative_on_measurement = False

    stable_measurement, measurement_poles = (
        measurement_derivative_controller.is_closed_loop_stable_discrete()
    )
    stable_error, error_poles = error_derivative_controller.is_closed_loop_stable_discrete()

    assert stable_measurement == stable_error
    assert np.allclose(np.sort_complex(measurement_poles), np.sort_complex(error_poles))


def test_unfiltered_discrete_stability_does_not_depend_on_tau():
    controllers = [_make_pid_controller(tau) for tau in (0.0, 0.24)]
    for controller in controllers:
        controller.sim.PID_filtered_derivative = False

    poles = [controller.is_closed_loop_stable_discrete()[1] for controller in controllers]

    assert np.allclose(np.sort_complex(poles[0]), np.sort_complex(poles[1]))


def test_discrete_stability_returns_result():
    controller = _make_pid_controller(tau=0.24)

    stable, poles = controller.is_closed_loop_stable_discrete()

    assert stable
    assert poles.size == controller.plant.dims + 4
    assert np.all(np.isfinite(poles))


@pytest.mark.parametrize(
    ('method', 'use_ode_mode'),
    [
        (IntegratorType.EULER_MARUYAMA, False),
        (IntegratorType.ANALYTIC_SDE, False),
        (IntegratorType.RK4, True),
        (IntegratorType.ANALYTIC_ODE, True),
    ],
)
def test_discrete_stability_supports_simulation_integrators(method, use_ode_mode):
    controller = _make_pid_controller(tau=0.24)
    controller.sim.integrator_method = method
    controller.sim.use_ode_mode = use_ode_mode

    stable, poles = controller.is_closed_loop_stable_discrete()

    assert isinstance(stable, bool)
    assert np.all(np.isfinite(poles))


def test_discrete_stability_ignores_inactive_pid_memory():
    controller = _make_pid_controller(tau=0.24)
    controller.sim.K_p = 20.0
    controller.sim.K_i = 0.0
    controller.sim.K_d = 0.0

    stable, poles = controller.is_closed_loop_stable_discrete()

    assert stable
    assert np.max(np.abs(poles)) < 1.0
