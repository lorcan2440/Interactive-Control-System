# pragma: no cover
if __name__ == "__main__":
    import __init__  # noqa

# external imports
import pytest
import numpy as np

# local imports
from plant import Plant


@pytest.fixture
def plant():
    """
    Provide a built valid Plant object for each test.

    This plant represents drug diffusion in a two-compartment model, with the following dynamics:

    - x_1' = -(k_12 + d) * x_1 + k_21 * x_2 + u + w_proc[0]
    - x_2' = k_12 * x_1 - (k_21 + d) * x_2 + w_proc[1]
    - y = x_2 + w_meas

    The meanings of the variables are:

    - x_1: drug concentration in compartment 1 (state variable 1)
    - x_2: drug concentration in compartment 2 (state variable 2)
    - u: drug injection rate in compartment 1 (control input)
    - w_proc: model disturbance in compartments 1 and 2
    - w_meas: measurement noise
    - y: measurement of compartment 2

    The constants are:

    - k_12, k_21: flow rates between compartments (defaults: k_12 = 10.0, k_21 = 20.0)
    - d: drug degradation rate (default: d = 1.0)

    and the noise matrices are:

    - Q = [[0.0, 0.0], [0.0, 1.0]]  (std.dev 1.0 in compartment 2, no noise in compartment 1)
    - R = [[1.0^2]]  (std.dev 1.0)
    """
    # kept for backward compatibility; prefer using `plant_factory`
    A = np.array([[-11.0, 20.0], [10.0, -21.0]])
    B = np.array([[1.0], [0.0]])
    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0]])

    Q = np.array([[0.0**2, 0.0], [0.0, 1.0**2]])  # noise only in compartment 2
    R = np.array([[1.0**2]])

    plant = Plant(dims=2, A=A, B=B, C=C, D=D, Q=Q, R=R)
    return plant


@pytest.fixture
def plant_factory():
    """
    Factory pattern fixture returning a function to create `Plant` objects
    with configurable `k_12`, `k_21`, and `d` parameters, as per the definition in `plant()`.

    Usage in tests:

    ```
    pf = plant_factory
    p_default = pf()  # sets to defaults: k_12 = 10, k_21 = 20, d = 1
    p_custom = pf(k_12=5, k_21=7, d=0.2)  # sets custom parameters
    ```
    """

    def _create(k_12=10, k_21=20, d=1, x_0=None, u_0=None):
        A = np.array([[-(k_12 + d), k_21], [k_12, -(k_21 + d)]])
        B = np.array([[1.0], [0.0]])
        C = np.array([[0.0, 1.0]])
        D = np.array([[0.0]])

        Q = np.array([[0.0**2, 0.0], [0.0, 1.0**2]])  # noise only in compartment 2
        R = np.array([[1.0**2]])

        plant = Plant(dims=2, x_0=x_0, u_0=u_0, A=A, B=B, C=C, D=D, Q=Q, R=R)
        return plant

    return _create
