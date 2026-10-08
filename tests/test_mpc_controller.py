import numpy as np

from controllers import KalmanFilter, ModelPredictiveController
from integrators import IntegratorType
from plant import Plant


class SimpleSim:
    def __init__(self):
        self.dt_anim = 0.1
        self.dt_int = 0.1
        self.integrator_method = IntegratorType.ANALYTIC_ODE
        self.use_ode_mode = True
        self.y_sp = np.array([[1.0]])
        self.MPC_N = 2


def make_plant():
    return Plant(
        dims=1,
        A=np.array([[-1.0]]),
        B=np.array([[1.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[0.0]]),
    )


def test_kalman_filter_corrects_prediction_with_measurement():
    observer = KalmanFilter(
        A=np.array([[1.0]]),
        B=np.array([[0.0]]),
        C=np.array([[1.0]]),
        D=np.array([[0.0]]),
        Q=np.array([[0.0]]),
        R=np.array([[1.0]]),
    )

    estimate = observer.update(np.array([[2.0]]), np.array([[0.0]]))

    assert estimate.shape == (1, 1)
    assert 0.0 < estimate[0, 0] < 2.0
    assert np.allclose(observer.P, observer.P.T)


def test_mpc_solves_qp_and_returns_finite_control():
    plant = make_plant()
    sim = SimpleSim()
    controller = ModelPredictiveController(simulator=sim, plant=plant)

    control = controller.calc_u(np.array([[1.0]]))

    assert control.shape == (1, 1)
    assert np.isfinite(control[0, 0])
    assert control[0, 0] > 0.0


def test_mpc_terminal_constraint_does_not_use_input_coefficients():
    controller = ModelPredictiveController(simulator=SimpleSim(), plant=make_plant())
    controller.ensure_horizon(2)
    controller.update_discrete_model()
    controller.constraints[2] = {
        'M': np.array([[1.0]]),
        'N': np.empty((0, 1)),
        'b': np.array([[3.0]]),
    }

    qp = controller._assemble_qp(
        initial_state_deviation=np.array([[0.0]]),
        x_ss=np.array([[0.0]]),
        u_ss=np.array([[0.0]]),
        horizon=2,
    )

    assert qp[2].shape[0] == 4
    assert qp[3].shape == (4,)
    assert qp[4].shape == (4,)


def test_mpc_terminal_constraint_accepts_empty_input_vector():
    controller = ModelPredictiveController(simulator=SimpleSim(), plant=make_plant())
    controller.ensure_horizon(2)
    controller.update_discrete_model()
    controller.constraints[2] = {
        'M': np.array([[1.0], [-1.0]]),
        'N': np.empty((0, 1)),
        'b': np.array([[3.0], [3.0]]),
    }

    qp = controller._assemble_qp(
        initial_state_deviation=np.array([[0.0]]),
        x_ss=np.array([[0.0]]),
        u_ss=np.array([[0.0]]),
        horizon=2,
    )

    assert qp[2].shape[0] == 5
    assert np.allclose(qp[4][-2:], [3.0, 3.0])


def test_mpc_horizon_expansion_copies_running_and_terminal_settings():
    controller = ModelPredictiveController(simulator=SimpleSim(), plant=make_plant())
    controller.costs[1] = {
        'V_xx': np.array([[2.0]]),
        'V_yy': 3.0,
        'V_uu': 4.0,
    }
    controller.costs[2] = {
        'V_xx': np.array([[5.0]]),
        'V_yy': 6.0,
        'V_uu': 7.0,
    }
    controller.constraints[1] = {
        'M': np.array([[1.0]]),
        'N': np.array([[2.0]]),
        'b': np.array([[3.0]]),
    }
    controller.constraints[2] = {
        'M': np.array([[-1.0]]),
        'N': np.empty((0, 1)),
        'b': np.array([[8.0]]),
    }

    controller.ensure_horizon(5)

    for stage in range(1, 5):
        assert np.array_equal(controller.costs[stage]['V_xx'], [[2.0]])
        assert controller.costs[stage]['V_yy'] == 3.0
        assert controller.costs[stage]['V_uu'] == 4.0
        assert np.array_equal(controller.constraints[stage]['M'], [[1.0]])
        assert np.array_equal(controller.constraints[stage]['N'], [[2.0]])
        assert np.array_equal(controller.constraints[stage]['b'], [[3.0]])

    assert np.array_equal(controller.costs[5]['V_xx'], [[5.0]])
    assert controller.costs[5]['V_yy'] == 6.0
    assert controller.costs[5]['V_uu'] == 7.0
    assert np.array_equal(controller.constraints[5]['M'], [[-1.0]])
    assert controller.constraints[5]['N'].shape == (0, 1)
    assert np.array_equal(controller.constraints[5]['b'], [[8.0]])

    # Expanded stages are independent copies, rather than aliases to one dict.
    controller.constraints[2]['M'][0, 0] = 99.0
    assert controller.constraints[3]['M'][0, 0] == 1.0


def test_mpc_resets_internal_model_and_weights_after_plant_dimension_change():
    plant = make_plant()
    controller = ModelPredictiveController(simulator=SimpleSim(), plant=plant)
    controller.update_discrete_model()

    plant.dims = 2
    plant.x_0 = np.zeros((2, 1))
    plant.x = plant.x_0.copy()
    plant.set_all_arrays(
        A=np.array([[-1.0, 0.0], [0.0, -2.0]]),
        B=np.array([[1.0], [0.0]]),
        C=np.array([[1.0, 0.0]]),
        D=np.array([[0.0]]),
        Q=np.zeros((2, 2)),
        R=np.array([[0.0]]),
    )

    controller.reset_for_state_dimension()

    assert controller.A_hat.shape == (2, 2)
    assert controller.costs[0]['V_xx'].shape == (2, 2)
    assert controller.observer is None
    controller.update_discrete_model()
    assert controller.observer.A.shape == (2, 2)
