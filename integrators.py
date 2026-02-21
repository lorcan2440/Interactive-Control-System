# external imports
from enum import Enum, auto

import numpy as np
from scipy.linalg import expm

# local imports
import plant
from utils import get_t_span, EPS


class IntegratorType(Enum):

    # for ODEs only (use_ode_mode = True):
    RK4 = auto()
    ANALYTIC_ODE = auto()

    # for SDEs only (use_ode_mode = False):
    EULER_MARUYAMA = auto()
    ANALYTIC_SDE = auto()

    def __str__(self):
        """Return the string representation for display purposes"""
        return self.name


################################################
## Integrators for ODEs (use_ode_mode = True) ##
################################################

def integrate_rk4(plant, t_start: float, t_stop: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the ODE: dx/dt = A x + B u + w_proc, where w_proc is a constant process noise term
    sampled at the start of the integration interval. This is not mathematically equivalent to
    solving the true SDE, since the noise is not treated as a stochastic diffusion term.

    This method uses the Runge-Kutta 4th order method (RK4) to get an approximate solution to the ODE, with time step dt.

    ### Arguments
    - `plant`: the plant to integrate
    - `t_start`: start time of the integration interval. Shape: scalar
    - `t_stop`: stop time of the integration interval. Shape: scalar
    - `dt`: time step for the output time points. Shape: scalar

    ### Returns
    - `tuple[np.ndarray, np.ndarray]`: arrays of time points and state trajectory over the integration interval 
    (including both endpoints).

    NOTE: RK4 assumes a smooth RHS. The process noise (modelled as Gaussian noise) is not 
    differentiable, so RK4 is not appropriate for use in stochastic differential equation (SDE) mode.
    """

    # create array [t_start, t_start + dt, t_start + 2 dt, ..., t_stop]
    t_span = get_t_span(t_start, t_stop, dt)  # shape (num_steps,)
    num_steps = t_span.shape[0]

    # set empty arrays
    x_span = np.zeros((plant.dims, num_steps))  # shape (dims, num_steps)

    # set initial state
    x_span[:, 0] = plant.x.reshape(plant.dims,)

    Bu = plant.B @ plant.u  # shape (dims, 1)

    # in ODE mode, we treat process noise as a constant drift term across the integration interval
    # sample the process noise once at the start of the interval from N(0, Q) and use it throughout the interval
    w_proc_i = plant.sample_process_noise(n=1)  # shape (dims, 1)

    for i in range(num_steps - 1):

        # get t_i
        t_i = t_span[i]
        dt_i = t_span[i + 1] - t_i
        x_i = x_span[:, i].reshape(plant.dims, 1)  # shape (dims, 1)

        # use Runge-Kutta 4th order method (RK4) with fixed step size dt
        # NOTE: dt may be different for the last step, so we use t_span[i + 1] - t_i instead of dt here
        # NOTE: the control input is constant across the frame, so we always use plant.u
        k1 = plant.A @ x_i + Bu + w_proc_i
        k2 = plant.A @ (x_i + 0.5 * k1 * dt_i) + Bu + w_proc_i
        k3 = plant.A @ (x_i + 0.5 * k2 * dt_i) + Bu + w_proc_i
        k4 = plant.A @ (x_i + k3 * dt_i) + Bu + w_proc_i
        x_dot_i = (k1 + 2 * k2 + 2 * k3 + k4) / 6  # x' at t_i, adjusted by RK4
        x_span[:, i + 1] = x_span[:, i] + x_dot_i.flatten() * dt_i  # shape (dims,)

    return t_span, x_span

def integrate_analytic_ode(plant, t_start: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the ODE: dx/dt = A x + B u + w_proc, where w_proc is a constant process noise term
    sampled at the start of the integration interval. This is not mathematically equivalent to
    solving the true SDE, since the noise is not treated as a stochastic diffusion term.
    
    This method uses the matrix exponential to compute the exact solution to the ODE, which is
    x(t) = e^{A t} @ x_0 + A^{-1} @ (e^{A t} - I) @ (B u + w_proc), where x_0 is the initial 
    state at the start of the integration interval.

    NOTE: the endpoint and step size is fixed, and is set in Plant.t_span_0.

    ### Arguments
    - `plant`: the plant to integrate
    - `t_start`: start time of the integration interval. Shape: scalar

    ### Returns
    - `tuple[np.ndarray, np.ndarray]`: arrays of time points and state trajectory over the integration interval 
    (including both endpoints).
    """
    t_span = plant.t_span_0 + t_start
    I = np.eye(plant.dims)

    # compute Phi
    Phi = plant.exp_A_t_span  # shape (num_steps, dims, dims)
    # compute B u + w_proc  (where w_proc is constant)
    Bu_plus_w = plant.B @ plant.u + plant.sample_process_noise(n=1)  # shape (dims, 1)
    # compute A^{-1} @ (e^{A t} - I) @ (B u + w_proc)
    Phi_plus = np.linalg.solve(plant.A, (Phi - I))  # shape (num_steps, dims, dims)
    # compute x using exact solution:
    # x = e^{A t} @ x_0 + A^{-1} @ (e^{A t} - I) @ (B u + w_proc)
    x_span = (Phi @ plant.x + Phi_plus @ Bu_plus_w)[:, :, 0].T  # shape (dims, num_steps)
    return t_span, x_span


#################################################
## Integrators for SDEs (use_ode_mode = False) ##
#################################################

# TODO: consider adding RK4 for Stratonovich SDEs, based on Frankignoul and Hasselmann (1976)
# source: https://github.com/bekaiser/SDE

def integrate_euler_maruyama(plant, t_start: float, t_stop: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the Itô SDE: dx = (A x + B u) dt + d_eta_t, where eta_t is a scaled Wiener process 
    given by eta_t = G @ W_t, where W_t is a standard Wiener process. These processes satisfy the Itô isometry
    (E[dW_t @ dW_t.T] = I dt and E[d_eta_t @ d_eta_t.T] = Q dt), so that the covariance matrix of the 
    diffusion term (process noise) is Q = G @ G.T.

    This method uses the Euler-Maruyama method to get an approximate solution to the SDE, with time step dt.

    ### Arguments
    - `plant`: the plant to integrate
    - `t_start`: start time of the integration interval. Shape: scalar
    - `t_stop`: stop time of the integration interval. Shape: scalar
    - `dt`: time step for the output time points. Shape: scalar

    ### Returns
    - `tuple[np.ndarray, np.ndarray]`: arrays of time points and state trajectory over the integration interval 
    (including both endpoints).

    NOTE: if needed, we can compute G using the Cholesky decomposition of Q, since Q is positive semidefinite. 
    However, drawing samples from N(0, Q) is more efficient (and mathemtically equivalent) than drawing 
    samples from N(0, I) and multiplying by G.
    """

    # create array [t_start, t_start + dt, t_start + 2 dt, ..., t_stop]
    t_span = get_t_span(t_start, t_stop, dt)  # shape (num_steps,)
    num_steps = t_span.shape[0]

    # set empty arrays
    x_span = np.zeros((plant.dims, num_steps))  # shape (dims, num_steps)

    # set initial state
    x_span[:, 0] = plant.x.reshape(plant.dims,)

    Bu = plant.B @ plant.u  # shape (dims, 1)

    # NOTE: draw from N(0, Q) for each time step to get diffusion innovations
    # this gets scaled by sqrt(dt) in the update loop to get the correct covariance for the diffusion term
    eta_proc = plant.sample_process_noise(n=num_steps - 1)  # shape (dims, num_steps - 1)

    for i in range(num_steps - 1):

        # get t_i
        t_i = t_span[i]
        dt_i = t_span[i + 1] - t_i
        x_i = x_span[:, i].reshape(plant.dims, 1)  # shape (dims, 1)

        eta_proc_i = eta_proc[:, i].reshape(plant.dims, 1)
        sqrt_dt = np.sqrt(dt_i)
        # Euler-Maruyama update:
        # x_{i+1} = x_i + (A x_i + B u) dt + eta_i sqrt(dt), where eta_i ~ N(0, Q)
        x_span[:, i + 1] = x_span[:, i] + (plant.A @ x_i + Bu)[:, 0] * dt_i + eta_proc_i[:, 0] * sqrt_dt

    return t_span, x_span

def integrate_analytic_sde(plant, t_start: float) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Solves the Itô SDE: dx = (A x + B u) dt + d_eta_t, where eta_t is a scaled Wiener process 
    given by eta_t = G @ W_t, where W_t is a standard Wiener process. These processes satisfy the Itô isometry
    (E[dW_t @ dW_t.T] = I dt and E[d_eta_t @ d_eta_t.T] = Q dt), so that the covariance matrix of the 
    diffusion term (process noise) is Q = G @ G.T.

    This method uses the Van Loan method to compute the exact discrete-time process noise
    covariance matrix Q_d over the integration interval, and samples noise from N(0, Q_d) to
    get an exact sample solution to the SDE.

    NOTE: the endpoint and step size is fixed, and is set in Plant.t_span_0.

    ### Arguments
    - `plant`: the plant to integrate
    - `t_start`: start time of the integration interval. Shape: scalar

    ### Returns
    - `tuple[np.ndarray, np.ndarray]`: arrays of time points and state trajectory over 
    the integration interval (including both endpoints).
    """

    t_span = plant.t_span_0 + t_start
    num_steps = t_span.shape[0]
    I = np.eye(plant.dims)

    # compute Van Loan block matrix
    # source: https://personales.upv.es/asala/DocenciaOnline/material/DiscretizRuidoVanLoanEN.pdf
    M = np.block([[plant.A,                  plant.Q   ],
                  [np.zeros_like(plant.A),   -plant.A.T]])  # shape (2 * dims, 2 * dims)
    M_exp = expm(M * plant.dt_int)  # shape (2 * dims, 2 * dims)
    Phi = M_exp[:plant.dims, :plant.dims]  # shape (dims, dims)
    S = M_exp[:plant.dims, plant.dims:]  # shape (dims, dims)
    Q_d = S @ Phi.T  # shape (dims, dims)

    # check Q_d is close to being symmetric (it should already be)
    if not np.allclose(Q_d, Q_d.T, atol=EPS):
        plant.logger.warning(f'''Integrated process noise covariance matrix Q_d is 
            not symmetric. Q_d: {Q_d}. Taking the symmetric component and continuing.''')

    # take the symmetric component (ideally no change to within numerical precision)
    Q_d = 0.5 * (Q_d + Q_d.T)

    Gamma = np.linalg.solve(plant.A, (Phi - I) @ plant.B)
    Gamma_u = (Gamma @ plant.u).flatten()

    # sample integrated process noise
    w_proc_int = np.random.multivariate_normal(mean=np.zeros(plant.dims), cov=Q_d,
        size=num_steps - 1, check_valid='raise').T

    # use state update equation: x_{i+1} = Phi @ x_i + Gamma @ u + w_proc_int
    # where w_proc_int ~ N(0, Q_d)
    x_span = np.zeros((plant.dims, num_steps))
    x_span[:, 0] = plant.x[:, 0]
    for i in range(num_steps - 1):
        x_span[:, i + 1] = Phi @ x_span[:, i] + Gamma_u + w_proc_int[:, i]

    return t_span, x_span
