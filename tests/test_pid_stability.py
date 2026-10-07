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
