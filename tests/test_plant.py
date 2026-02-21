# external imports
import pytest
import numpy as np
import time

# local imports
if __name__ == '__main__':
    import __init__
from plant import Plant


# set fixed seed for reproducibility of noise sampling
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(0)


def test_init_linear_plant(plant_factory):

    # test valid init
    A3 = np.eye(3) * -1.0
    B3 = np.zeros((3, 1))
    C3 = np.zeros((1, 3))
    D3 = np.zeros((1, 1))
    plant = Plant(dims=3, x_0=np.zeros((3, 1)), u_0=np.zeros((1, 1)), A=A3, B=B3, C=C3, D=D3)
    assert plant.dims == 3
    assert np.array_equal(plant.u, np.zeros((1, 1)))
    assert np.array_equal(plant.x, np.zeros((3, 1)))

    # test invalid init
    with pytest.raises(ValueError):
        plant = Plant(dims=2.5)

    # test plant factory custom setting works
    plant = plant_factory(k_12=5, k_21=7, d=0.2)
    assert np.array_equal(plant.A, np.array([[-5.2, 7], [5, -7.2]]))


def test_set_state_space_matrices(plant_factory):

    # init
    plant = plant_factory()

    # test valid matrices
    A = np.array([[-1, 0], [0, -2]])
    B = np.array([[1], [0]])
    C = np.array([[0, 1]])
    D = np.array([[0]])

    plant.set_state_space_matrices(A, B, C, D)
    assert np.array_equal(plant.A, A)
    assert np.array_equal(plant.B, B)
    assert np.array_equal(plant.C, C)
    assert np.array_equal(plant.D, D)

    # test invalid matrices
    with pytest.raises(ValueError):
        plant.set_state_space_matrices(A=np.array([[-1, 0], [0, -2]]), 
                                       B=np.array([[1], [0]]), 
                                       C=np.array([[0, 1]]), 
                                       D=np.array([[0, 0]]))  # D has wrong shape


def test_set_noise_covariances(plant_factory):

    # init
    plant = plant_factory()

    # test defaults
    plant.set_noise_covariances()
    assert np.array_equal(plant.Q, np.zeros((2, 2)))
    assert np.array_equal(plant.R, np.zeros((1, 1)))

    # test valid covariances
    Q = np.array([[1.0, 0.5], [0.5, 2.0]])
    R = np.array([[2.0]])
    plant.set_noise_covariances(Q=Q, R=R)
    assert np.array_equal(plant.Q, Q)
    assert np.array_equal(plant.R, R)

    # test invalid noise covariances - R has wrong shape
    with pytest.raises(ValueError):
        plant.set_noise_covariances(R=np.array([[1.0, 0.5], [0.5, 1.0]]))
    
    # test invalid noise covariances - Q is not symmetric
    with pytest.raises(ValueError):
        plant.set_noise_covariances(Q=np.array([[1.0, 0.5], [0.0, 1.0]]))

    # test invalid noise covariances - Q is not positive semidefinite
    with pytest.raises(ValueError):
        plant.set_noise_covariances(Q=np.array([[0.05, 0.1], [0.1, 0.05]]))


def test_sample_noise(plant_factory):

    # create plant from factory with default params
    plant = plant_factory()

    # sample noise three times
    num_samples = 3
    w_proc = plant.sample_process_noise(n=num_samples)  # shape (dims, num_samples)
    w_meas = plant.sample_measurement_noise(n=num_samples)  # shape (1, num_samples)

    # check shapes
    assert w_proc.shape == (2, num_samples)
    assert w_meas.shape == (1, num_samples)

    # the factory default provides a plant with nonzero noise covariances
    # process noise has zero variance in compartment 1
    assert np.allclose(w_proc[0, :], 0.0)
    # process noise has nonzero variance in compartment 2
    assert not np.array_equal(w_proc[1, :], np.zeros(num_samples))
    # measurement noise has nonzero variance
    assert not np.array_equal(w_meas, np.zeros((1, num_samples)))


def test_dynamics(plant_factory):

    dt = 0.01
    n = 1000

    # Case 1: relaxation response, no process noise
    plant = plant_factory(x_0=np.array([[1.0], [1.0]]), u_0=np.array([[0.0]]))
    plant.set_noise_covariances(Q=np.zeros((2, 2)), R=np.zeros((1, 1)))  # no noise
    t_span, x_span = plant.integrate_dynamics(t_start=0.0, t_stop=(n - 1) * dt, dt=dt)
    assert t_span.shape == (n,)
    assert x_span.shape == (2, n)
    assert np.array_equal(x_span[:, 0], np.array([1.0, 1.0]))  # check initial state
    assert np.allclose(x_span[:, -1], np.array([0.0, 0.0]), atol=1e-2)  # check final state (near zero)

    # Case 2: step response, no process noise
    plant = plant_factory(x_0=np.array([[0.0], [0.0]]), u_0=np.array([[3.1]]))
    plant.set_noise_covariances(Q=np.zeros((2, 2)), R=np.zeros((1, 1)))  # no noise
    t_span, x_span = plant.integrate_dynamics(t_start=0.0, t_stop=(n - 1) * dt, dt=dt)
    assert np.array_equal(x_span[:, 0], np.array([0.0, 0.0]))  # check initial state
    assert np.allclose(x_span[:, -1], np.array([2.1, 1.0]), atol=1e-2)  # check final state (near [2.1, 1.0])

    # Case 3: step response with process noise, allowed to vary
    plant = plant_factory(x_0=np.array([[0.0], [0.0]]), u_0=np.array([[3.1]]))
    plant.set_noise_covariances(Q=np.array([[0.0, 0.0], [0.0, 0.3**2]]))
    t_span, x_span = plant.integrate_dynamics(t_start=0.0, t_stop=(n - 1) * dt, dt=dt, hold_noise_const=False)
    assert np.array_equal(x_span[:, 0], np.array([0.0, 0.0]))  # check initial state
    assert np.allclose(x_span[:, -1], np.array([2.1, 1.0]), atol=0.5)  # check final state (near [2.1, 1.0]), bigger tolerance

    # Case 4: step response with process noise, held constant
    plant = plant_factory(x_0=np.array([[0.0], [0.0]]), u_0=np.array([[3.1]]))
    plant.set_noise_covariances(Q=np.array([[0.0, 0.0], [0.0, 0.3**2]]))
    t_span, x_span = plant.integrate_dynamics(t_start=0.0, t_stop=(n - 1) * dt, dt=dt, hold_noise_const=True)
    assert np.array_equal(x_span[:, 0], np.array([0.0, 0.0]))  # check initial state
    assert np.allclose(x_span[:, -1], np.array([2.1, 1.0]), atol=0.5)  # check final state (near [2.1, 1.0]), bigger tolerance


def test_integrate_dynamics_speed_comparison(plant_factory):

    # keep horizon at one animation frame for simplicity
    t_start = 0.0
    t_stop = 1 / 60
    dt = 0.001
    warmup = 20
    repeats = 1000

    def measure(method: str) -> float:
        plant = plant_factory(x_0=np.array([[0.0], [0.0]]), u_0=np.array([[3.1]]))
        plant.set_noise_covariances(Q=np.zeros((2, 2)), R=np.zeros((1, 1)))

        # warm-up to reduce one-time overhead in the measured loop
        for _ in range(warmup):
            plant.x = plant.x_0.copy()
            plant.integrate_dynamics(t_start=t_start, t_stop=t_stop, dt=dt, method=method,
                hold_noise_const=True)

        # start timing
        t0 = time.perf_counter()
        for _ in range(repeats):
            plant.x = plant.x_0.copy()
            plant.integrate_dynamics(t_start=t_start, t_stop=t_stop, dt=dt, method=method,
                hold_noise_const=True)
        return time.perf_counter() - t0

    numerical_time = measure('numerical')
    analytic_time = measure('analytic')

    # use pytest -s to view this manually
    print(f'Numerical time: {numerical_time:.6f}s, Analytic time: {analytic_time:.6f}s')

    assert analytic_time < numerical_time, (
        'Expected analytic integration to be faster than numerical integration. '
        f'numerical={numerical_time:.6f}s, analytic={analytic_time:.6f}s'
    )
