# built-ins
import copy
from enum import Enum, auto

# external imports
import numpy as np
from scipy.linalg import block_diag, expm
from scipy import sparse
from scipy.linalg import solve_continuous_are

# local imports
from integrators import IntegratorType
from utils import get_logger, EPS, get_t_span, GUI_SLIDER_CONFIG


def discretise_state_space(A, B, integration_steps, method):
    """Discretise a continuous state-space model using the simulator's step method."""
    state_count = A.shape[0]
    augmented_matrix = np.block([
        [A, B],
        [np.zeros((B.shape[1], state_count + B.shape[1]))],
    ])
    identity = np.eye(augmented_matrix.shape[0])
    augmented_step = identity.copy()

    if method in (IntegratorType.ANALYTIC_ODE, IntegratorType.ANALYTIC_SDE):
        augmented_step = expm(augmented_matrix * sum(integration_steps))
    else:
        for step in integration_steps:
            scaled_matrix = augmented_matrix * step
            if method is IntegratorType.EULER_MARUYAMA:
                substep_matrix = identity + scaled_matrix
            elif method is IntegratorType.RK4:
                matrix_squared = scaled_matrix @ scaled_matrix
                matrix_cubed = matrix_squared @ scaled_matrix
                substep_matrix = (
                    identity + scaled_matrix + matrix_squared / 2
                    + matrix_cubed / 6 + matrix_cubed @ scaled_matrix / 24
                )
            else:
                raise ValueError(f'Unsupported integration method for MPC: {method}.')
            augmented_step = substep_matrix @ augmented_step

    return augmented_step[:state_count, :state_count], augmented_step[:state_count, state_count:]


def discretise_process_noise(A, Q, integration_steps, method, use_ode_mode):
    """Discretise process-noise covariance for the simulator's ODE or SDE semantics."""
    state_count = A.shape[0]
    if use_ode_mode:
        _, noise_gain = discretise_state_space(
            A, np.eye(state_count), integration_steps, method
        )
        return noise_gain @ Q @ noise_gain.T

    if method is IntegratorType.ANALYTIC_SDE:
        van_loan_matrix = np.block([
            [A, Q],
            [np.zeros_like(A), -A.T],
        ])
        van_loan_step = expm(van_loan_matrix * sum(integration_steps))
        transition = van_loan_step[:state_count, :state_count]
        covariance = van_loan_step[:state_count, state_count:] @ transition.T
        return 0.5 * (covariance + covariance.T)

    covariance = np.zeros_like(Q)
    for step in integration_steps:
        transition = np.eye(state_count) + A * step
        covariance = transition @ covariance @ transition.T + Q * step
    return 0.5 * (covariance + covariance.T)


class ControllerType(Enum):

    NONE = auto()
    MANUAL = auto()
    OPENLOOP = auto()
    BANGBANG = auto()
    PID = auto()
    H2 = auto()
    HINF = auto()
    MPC = auto()
    RL = auto()
    
    def __str__(self):
        """Return the string representation for display purposes"""
        return self.name


class ManualController:
    def __init__(self, sim: object = None, plant: object = None):
        '''
        The manual controller allows the user to directly specify the control input.
        No computations are performed; the controller simply returns the GUI slider `manual_u` value.
        '''
        self.sim = sim
        self.plant = plant
        self.logger = get_logger()

    def calc_u(self) -> np.ndarray:
        '''
        Calculate the control input for manual control. Ignores measurements and returns
        the GUI slider `manual_u`.

        ### Returns
        - `u`: control input. Shape: (1, 1)
        '''
        return np.array([[float(self.sim.manual_u)]])


class OpenLoopController:
    def __init__(self, sim: object = None, plant: object = None):
        '''
        An open-loop controller (aka feedforward controller) is a simple control law whose control input
        is proportional to the reciprocal of the steady-state gain of the plant.
        
        There is no measuring of the output i.e. no feedback, the block diagram is 'open': 
        the control input depends only on the setpoint.
        The controller response time is determined solely by the dynamics of the plant.
        In the complete absence of disturbances, the steady-state error will be zero.
        '''
        self.sim = sim
        self.plant = plant
        self.logger = get_logger()

    def calc_u(self) -> np.ndarray:
        '''
        Calculates the control input for an open-loop (feedforward) controller.
        This depends only on the current setpoint.

        NOTE: if the plant's A matrix is singular (has an eigenvalue of zero), then
        the control input will always be zero. This is because there is no
        finite step input that can produce a steady-state value. 
        
        In theory, we could apply an impulse input for one frame, using the formula:
        `u = e / (self.sim.dt_frame * C @ B)`, but this requires knowing the error (and hence measurement), 
        which is not allowed for an open-loop controller. Therefore, we choose not to implement this case
        and instead take u = 0.

        ### Returns
        - `u`: control input. Shape: (1, 1)
        '''

        # get setpoint
        y_sp = self.sim.y_sp

        # compute control input
        u = y_sp / self.plant.G_0
        return u


class BangBangController:
    def __init__(self, sim: object = None, plant: object = None):
        '''
        A bang-bang controller (aka 'on-off controller') only has two possible control inputs:

        - u = U_plus,  if y_measured < y_setpoint
        - u = U_minus, if y_measured > y_setpoint

        where U_plus > 0 and U_minus < 0 are constants and are the parameters of the controller.

        This type of controller resembles how a thermostat works. It can lead to chattering (rapid changes of u) 
        near the setpoint if there is measurement noise and/or if the plant process dynamics are fast.
        '''
        self.sim = sim
        self.plant = plant
        self.logger = get_logger()

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        '''
        Calculates the control input for a bang-bang (on-off) controller.
        Depending on the sign of `e`, one of two discrete control inputs are chosen.
        
        ### Arguments
        - `e`: error (difference in y: setpoint minus measurement). Shape: (1, 1)

        ### Returns
        - `u`: control input. Shape: (1, 1)
        '''

        # compute control input
        if e > 0:
            u = np.array([[self.sim.U_plus]])
        elif e < 0:
            u = np.array([[self.sim.U_minus]])
        else:
            u = np.array([[0.0]])
        return u


class RLController:
    """Run a trained continuous-action reinforcement-learning policy."""

    def __init__(self):
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        if self.agent is None:
            raise RuntimeError('Train or load an RL policy before selecting the RL controller.')
        if not isinstance(e, np.ndarray) or e.shape != (1, 1):
            raise ValueError('e must have shape (1, 1)')
        action = self.agent.select_action(e.reshape(-1))
        action = np.clip(
            action,
            self.agent.config.action_min,
            self.agent.config.action_max,
        )
        return np.asarray(action, dtype=float).reshape(1, 1)


class PIDController:

    # TODO: add anti-windup for integral term
    # TODO: add function to calculate PID parameters based on integrated absolute error (IAE) optimality
    # TODO: add function to calculate PID parameters based on integrated time-weighted absolute error (ITAE) optimality
    # TODO: add function to calculate PID parameters based on Ziegler-Nichols tuning rules
    # TODO: add function to calculate PID parameters based on Cohen-Coon tuning rules
    # TODO: add function to calculate PID parameters based on pole placement, given n closed-loop poles
    # TODO: add function to calculate gain and phase (unwrapped using first-order terms) of the OLTF at 
    # given frequency, and calculate the gain and phase margin

    def __init__(self, sim: object = None, plant: object = None):
        self.sim = sim
        self.plant = plant
        self.logger = get_logger()
        self.reset_memory()

    def reset_memory(self):
        '''
        Resets the internal state of the PID controller: accumulated error used in the integral term, 
        and previous derivative input and derivative action.
        '''
        self.e_integrated = np.array([[0.0]])
        self.derivative_input_prev = np.array([[0.0]])
        self.u_d_prev = np.array([[0.0]])
        self.cl_stable_prev = True

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        """
        Compute PID control input with optional derivative filtering and derivative-on-measurement.
        The filter is used to avoid excessive noise amplification from the derivative term. The frequency cutoff
        of the low-pass filter can be set by the (reciprocal of) the time constant `tau` in the GUI.

        ### Arguments
        - `e`: error (difference in y: setpoint minus measurement). Shape: (1, 1)

        ### Returns
        - `u`: control input. Shape: (1, 1)
        """

        # check that e has correct shape
        if not isinstance(e, np.ndarray) or e.shape != (1, 1):
            raise ValueError("e must have shape (1, 1)")
        
        # check closed-loop stability
        cl_stable, cl_z_poles = self.is_closed_loop_stable_discrete()
        if not cl_stable and self.cl_stable_prev:  # only log one warning
            self.logger.warning(f'''Closed-loop unstable for current PID parameters: 
                eigenvalues of A_cl (poles in z-plane) are {cl_z_poles}. Prev: {self.cl_stable_prev}''')
        elif cl_stable:
            self.cl_stable_prev = True
        else:
            self.cl_stable_prev = False

        # The derivative input is either the error (setpoint changes cause derivative action)
        # or the negative measurement (setpoint changes do not cause derivative action).
        y_meas = self.sim.y_sp - e
        derivative_on_measurement = getattr(self.sim, 'PID_derivative_on_measurement', True)
        derivative_input = -y_meas if derivative_on_measurement else e

        # sampling period
        dt = self.sim.dt_anim

        # controller parameters and time constants
        self.K_p = self.sim.K_p  # proportional gain
        self.K_i = self.sim.K_i  # integral gain
        self.K_d = self.sim.K_d  # derivative gain
        self.tau = self.sim.tau  # derivative filter time constant
        self.T_i = self.sim.K_p / self.sim.K_i if self.sim.K_i != 0 else np.inf  # integral time constant
        self.T_d = self.sim.K_d / self.sim.K_p if self.sim.K_p != 0 else np.inf  # derivative time constant

        # proportional term
        u_p = self.K_p * e

        # integral term
        self.e_integrated += e * dt
        u_i = self.K_i * self.e_integrated

        self.filtered_derivative = getattr(self.sim, 'PID_filtered_derivative', True)
        self.derivative_on_measurement = derivative_on_measurement

        # Derivative on either error or negative measurement. The minus sign for
        # measurement derivative gives the same feedback action as differentiating error.
        if self.K_d == 0:
            u_d = np.array([[0.0]])
        elif not self.filtered_derivative:
            u_d = self.K_d * (derivative_input - self.derivative_input_prev) / dt
        else:
            # low-pass filter time constant: use user-configured `tau` when available,
            # otherwise fall back to 5x the sampling period
            # NOTE: consider setting this to 0.1x the derivative time constant K_p / K_d
            tau = getattr(self, 'tau', max(5.0 * dt, 1e-6))

            # Discrete first-order low-pass filter on the derivative term.
            alpha = 1.0 - dt / tau
            alpha = max(min(alpha, 1.0), 0.0)

            u_d = alpha * self.u_d_prev + (self.K_d / tau) * (
                derivative_input - self.derivative_input_prev
            )
            self.u_d_prev = u_d

        self.derivative_input_prev = derivative_input.copy()

        # total control input = P + I + D
        u = u_p + u_i + u_d

        return u
    
    def K_y(self, s: complex) -> complex:
        '''
        Controller transfer function (continuous-time) from y_meas to u.
        '''
        derivative_tf = self.K_d * s
        if getattr(self.sim, 'PID_filtered_derivative', True):
            derivative_tf /= self.tau * s + 1
        return -self.K_p - self.K_i / s - derivative_tf
    
    def K_sp(self, s: complex) -> complex:
        '''
        Controller transfer function (continuous-time) from y_sp to u.
        '''
        derivative_tf = 0.0
        if not getattr(self.sim, 'PID_derivative_on_measurement', True):
            derivative_tf = self.K_d * s
            if getattr(self.sim, 'PID_filtered_derivative', True):
                derivative_tf /= self.tau * s + 1
        return self.K_p + self.K_i / s + derivative_tf
    
    def K(self, s: complex) -> np.ndarray:

        '''
        Controller transfer function (continuous-time) from [y_meas, y_sp]^T to u, 
        suitable for use in the generalised plant and controller interconnection.

        The first entry is the TF from y_meas to u. The second entry is the TF from y_sp to u.

        The derivative filter and derivative input are configured by the PID options.
        '''

        return np.array([[self.K_y(s), self.K_sp(s)]])  # shape: (1, 2)
    
    def lower_LFT(self, s: complex, C: float = 1.0) -> np.ndarray:
        '''
        Evaluate the lower linear fractional transformation (LFT) F_l(P, K)(s) of the
        generalised plant P(s) and the controller K(s).

        The lower LFT is the transfer function from the generalised disturbance input w = [d_i, d_o, y_sp]^T
        to the performance output z = e + C u, where e is the error (y_sp - y_meas) and C is the 
        performance gain on the control input u.
        '''

        # plant TFs
        G_p = self.plant.G_p(s)  # TF from u to y_meas
        G_d = self.plant.G_d(s)  # TF from d_i to y_meas

        # controller TFs
        K_y = self.K_y(s)  # TF from y_meas to u
        K_sp = self.K_sp(s)  # TF from y_sp to u

        # compute lower LFT using formula: F_l(P, K) = P11 + P12 @ K @ (I - P22 @ K)^(-1) @ P21
        L = (C - G_p) / (1 - K_y * G_p)
        return np.array([
            [-G_d + L * K_y * G_d, -1 + L * K_y, 1 + L * K_sp]
        ])
    
    def is_closed_loop_stable_discrete(self) -> tuple[bool, np.ndarray]:
        """Check asymptotic stability of the sampled PID loop used by the simulator.

        The closed-loop state is ``[x_plant, x_controller, u_previous]``.
        Its poles are discrete-time eigenvalues, so the loop is stable exactly
        when every pole has magnitude less than one.
        """

        # get the plant matrices
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D

        # get PID controller params
        K_p, K_i, K_d, tau = float(self.sim.K_p), float(self.sim.K_i), float(self.sim.K_d), float(self.sim.tau)
        dt_anim = float(self.sim.dt_anim)
        filtered_derivative = getattr(self.sim, 'PID_filtered_derivative', True)
        if filtered_derivative and tau <= 0:
            raise ValueError(f'PID derivative filter time constant must be > 0 (got {tau}).')

        integration_method = self.sim.integrator_method
        use_ode_mode = self.sim.use_ode_mode

        # augment the continuous plant with a constant input state:
        # d/dt [x, u] = [[A, B], [0, 0]] @ [x, u].
        A_aug = np.block([[A, B], [np.zeros((1, A.shape[1] + 1))]])
        A_aug_step = np.eye(A_aug.shape[0])
        integration_steps = np.diff(self.plant.t_span_0)

        if use_ode_mode and integration_method not in (IntegratorType.RK4, IntegratorType.ANALYTIC_ODE):
            raise ValueError(f'Invalid integration method for ODE mode: {integration_method}.')
        if not use_ode_mode and integration_method not in (IntegratorType.EULER_MARUYAMA, IntegratorType.ANALYTIC_SDE):
            raise ValueError(f'Invalid integration method for SDE mode: {integration_method}.')

        if integration_method in (IntegratorType.ANALYTIC_ODE, IntegratorType.ANALYTIC_SDE):
            A_aug_step = expm(A_aug * float(np.sum(integration_steps)))
        else:
            I_aug = np.eye(A_aug.shape[0])
            for step in integration_steps:
                A_step_scaled = A_aug * step
                if integration_method is IntegratorType.EULER_MARUYAMA:
                    A_plant_substep_augmented = I_aug + A_step_scaled
                else:
                    A_plant_step_squared = A_step_scaled @ A_step_scaled
                    A_plant_step_cubed = A_plant_step_squared @ A_step_scaled
                    # use a 4th-order Taylor series expansion of the matrix exponential for RK4 integration.
                    A_plant_substep_augmented = (
                        I_aug + A_step_scaled
                        + A_plant_step_squared / 2
                        + A_plant_step_cubed / 6
                        + (A_plant_step_cubed @ A_step_scaled) / 24
                    )
                A_aug_step = A_plant_substep_augmented @ A_aug_step

        # get discrete-time plant matrices
        A_d = A_aug_step[:-1, :-1]
        B_d = A_aug_step[:-1, -1:]

        # get discrete-time state space realisation of the PID controller
        # controller state: [integral_error, previous_measurement, derivative_output]
        alpha = max(min(1.0 - dt_anim / tau, 1.0), 0.0) if filtered_derivative else 0.0
        A_Kd = np.zeros((3, 3))
        B_Kd = np.array([[0.0], [1.0], [0.0]])
        if K_i != 0:
            A_Kd[0, 0] = 1.0
            B_Kd[0, 0] = -dt_anim
        if K_d != 0:
            derivative_scale = K_d / tau if filtered_derivative else K_d / dt_anim
            A_Kd[2, 1:] = [derivative_scale, alpha]
            B_Kd[2, 0] = -derivative_scale
        derivative_scale = K_d / tau if filtered_derivative else K_d / dt_anim
        C_Kd = np.array([[K_i, derivative_scale, alpha]])
        D_Kd = np.array([[-K_p - K_i * dt_anim - derivative_scale,]])

        # get closed-loop discrete-time state space A-matrix
        A_cl = np.block([
            [A_d + B_d @ D_Kd @ C, B_d @ C_Kd, B_d @ D_Kd @ D],
            [B_Kd @ C, A_Kd, B_Kd @ D],
            [D_Kd @ C, C_Kd, D_Kd @ D]
        ])

        # check whether all eigenvalues of A_cl lie within the unit circle (discrete-time stability condition)
        poles = np.linalg.eigvals(A_cl)
        stable = bool(np.all(np.abs(poles) < 1.0))

        return stable, poles



class H2Controller:
    """Continuous-time LQG controller with state cost ``(C1 @ x)**2``."""

    def __init__(self, simulator, plant):
        self.simulator = simulator
        self.plant = plant
        self.logger = get_logger()
        self._design_key_prev = None
        self.cl_stable_prev = True
        self.reset_memory()

    def reset_memory(self):
        """Reset the estimated state to the plant's current state."""
        self.x_hat = self.plant.x.copy()
        self._has_previous_sample = False

    def update_controller_matrices(self):
        """Update the H2 controller state space matrices if the plant or controller parameters have changed."""

        # get the plant state space matrices
        dims = self.plant.dims
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        Q = self.plant.Q
        R = np.maximum(self.plant.R, np.array([[EPS]]))  # minimum measurement noise variance to avoid singularity
        R_item = R.item()

        # get controller parameters from the GUI
        C1_x = np.array([
            float(getattr(self.simulator, f'H2_C1_x{i + 1}', 1.0))
            for i in range(dims)
        ]).reshape(1, dims)
        C1_u = np.array([[float(getattr(self.simulator, 'H2_C1_u', 1.0))]])

        # create a hashable key for the current design matrices and parameters to avoid unnecessary recomputation
        design_matrices = (A, B, C, D, Q, R, C1_x, C1_u)
        design_key = (self.simulator.dt_anim, *((matrix.shape, matrix.dtype.str, matrix.tobytes()) for matrix in design_matrices),)

        if design_key == self._design_key_prev:
            return

        C1_x_squared = C1_x.T @ C1_x
        C1_u_squared = C1_u.T @ C1_u
        try:
            # solve the continuous algebraic Riccati equation (CARE) for the state cost X
            X = solve_continuous_are(A, B, C1_x_squared, C1_u_squared)
            self.F = B.T @ X

            # solve the filter algebraic Riccati equation (FARE) for the measurement cost Y
            Y = solve_continuous_are(A.T, C.T, Q, R)
            self.H = Y @ C.T / R_item

        except (ValueError, np.linalg.LinAlgError) as error:
            self.logger.exception('Unable to solve H2/LQG Riccati equations.')
            raise ValueError(
                'H2 controller design failed; check that the plant is stabilisable '
                'and detectable and that its noise matrices are valid.'
            ) from error

        # exact sampled transition for the continuous-time observer driven by
        # held plant input and the most recent measured output.
        A_K = A - self.H @ C
        B_K = B - self.H @ D

        # compute the discrete-time observer matrices by augmenting the observer with a constant input state
        K_aug = np.zeros((dims + 2, dims + 2))
        K_aug[:dims, :dims] = A_K
        K_aug[:dims, dims:dims + 1] = B_K
        K_aug[:dims, dims + 1:] = self.H

        # compute the exact discrete-time transition matrix for the augmented observer
        # and get the discrete-time observer matrices from it
        K_step = expm(K_aug * self.simulator.dt_anim)
        self.A_Kd = K_step[:dims, :dims]
        self.B_Kd = K_step[:dims, dims:dims + 1]
        self.H_Kd = K_step[:dims, dims + 1:]

        self._design_key_prev = design_key
        self.logger.debug(f'H2 design updated: dt_anim={self.simulator.dt_anim}, C1_x={C1_x}, C1_u={C1_u}, '
                          f'F={self.F}, H={self.H}, observer poles z={np.linalg.eigvals(A_K)}')

    def is_closed_loop_stable_discrete(self) -> tuple[bool, np.ndarray]:
        """Check stability of the sampled plant and H2 controller interconnection."""
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        dims = self.plant.dims
        dt_anim = float(self.simulator.dt_anim)
        integration_method = self.simulator.integrator_method
        use_ode_mode = self.simulator.use_ode_mode

        if use_ode_mode and integration_method not in (IntegratorType.RK4, IntegratorType.ANALYTIC_ODE):
            raise ValueError(f'Invalid integration method for ODE mode: {integration_method}.')
        if not use_ode_mode and integration_method not in (IntegratorType.EULER_MARUYAMA, IntegratorType.ANALYTIC_SDE):
            raise ValueError(f'Invalid integration method for SDE mode: {integration_method}.')

        # discretise the deterministic plant dynamics using the same method and
        # substeps as the simulator; noise does not affect asymptotic stability
        A_aug = np.block([[A, B], [np.zeros((1, dims + 1))]])
        I_aug = np.eye(dims + 1)
        A_aug_step = I_aug.copy()
        integration_steps = np.diff(get_t_span(0.0, dt_anim, self.plant.dt_int))

        if integration_method in (IntegratorType.ANALYTIC_ODE, IntegratorType.ANALYTIC_SDE):
            A_aug_step = expm(A_aug * dt_anim)
        else:
            for step in integration_steps:
                A_step_scaled = A_aug * step
                if integration_method is IntegratorType.EULER_MARUYAMA:
                    A_plant_step = I_aug + A_step_scaled
                else:
                    A_step_squared = A_step_scaled @ A_step_scaled
                    A_step_cubed = A_step_squared @ A_step_scaled
                    A_plant_step = (
                        I_aug + A_step_scaled
                        + A_step_squared / 2
                        + A_step_cubed / 6
                        + (A_step_cubed @ A_step_scaled) / 24
                    )
                A_aug_step = A_plant_step @ A_aug_step

        A_d = A_aug_step[:dims, :dims]
        B_d = A_aug_step[:dims, dims:]

        # State is [x_plant, x_hat_previous, u_previous]. The observer first
        # estimates state from the current output; state feedback then sets the
        # input held over the next plant frame.
        observer_input = self.B_Kd + self.H_Kd @ D
        A_cl = np.block([
            [A_d - B_d @ self.F @ self.H_Kd @ C,
             -B_d @ self.F @ self.A_Kd,
             -B_d @ self.F @ observer_input],
            [self.H_Kd @ C, self.A_Kd, observer_input],
            [-self.F @ self.H_Kd @ C,
             -self.F @ self.A_Kd,
             -self.F @ observer_input],
        ])

        poles = np.linalg.eigvals(A_cl)
        stable = bool(np.all(np.abs(poles) < 1.0))
        return stable, poles

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        """Calculate a setpoint-tracking control input with shape ``(1, 1)``."""
        if not isinstance(e, np.ndarray) or e.shape != (1, 1):
            raise ValueError('e must have shape (1, 1)')

        self.update_controller_matrices()
        cl_stable, cl_z_poles = self.is_closed_loop_stable_discrete()
        if not cl_stable and self.cl_stable_prev:
            self.logger.warning(f'Closed-loop unstable: eigenvalues of A_cl (poles in z-plane) are {cl_z_poles}.')
        self.cl_stable_prev = cl_stable

        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        dims = self.plant.dims
        if self.x_hat.shape != (dims, 1):
            self.reset_memory()

        # solve to get the equilibrium state and input for the current setpoint
        # (x_ss and u_ss such that C @ x_ss + D @ u_ss = y_sp and A @ x_ss + B @ u_ss = 0)
        A_eq = np.block([[A, B], [C, D]])
        b_eq = np.vstack([np.zeros((dims, 1)), self.simulator.y_sp])
        try:
            x_u_ss = np.linalg.solve(A_eq, b_eq)
        except np.linalg.LinAlgError as error:
            self.logger.error('H2 controller cannot track the setpoint: equilibrium matrix is singular.')
            raise ValueError('H2 controller cannot track this setpoint because the plant has no unique steady-state solution.') from error
        x_ss, u_ss = x_u_ss[:dims], x_u_ss[dims:]

        # observer update: estimate x based on the previous x_hat, held u, and current y_meas
        y_meas = self.simulator.y_sp - e  # reconstruct current measurement
        if self._has_previous_sample:
            self.x_hat = self.A_Kd @ self.x_hat + self.B_Kd @ self.plant.u + self.H_Kd @ y_meas
        else:
            self._has_previous_sample = True

        # compute u based on x_hat, shifted by the steady-state values (controller is a regulator: drives delta_x and delta_u to zero)
        delta_x = self.x_hat - x_ss
        delta_u = -self.F @ delta_x
        u = u_ss + delta_u
        return u


class HInfinityController:
    """Continuous-time output-feedback H-infinity controller.

    The performance output is the stack ``[C1_x @ x, C1_u * u]``. The
    CARE/FARE design finds the smallest feasible gamma by bisection.
    """

    def __init__(self, simulator, plant):
        self.simulator = simulator
        self.plant = plant
        self.logger = get_logger()
        self._design_key = None
        self.cl_stable_prev = True
        self.reset_memory()

    def reset_memory(self):
        """Reset the dynamic controller state."""
        self.x_controller = np.zeros((self.plant.dims, 1))

    def _get_design_data(self):
        dims = self.plant.dims
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        C1_x = np.array([float(getattr(self.simulator, f'Hinf_C1_x{i + 1}', 1.0)) for i in range(dims)]).reshape(1, dims)
        C1_u = float(getattr(self.simulator, 'Hinf_C1_u', 1.0))
        if C1_u <= 0.0:
            raise ValueError('H-infinity input weight Hinf_C1_u must be greater than zero.')

        process_noise, measurement_noise = self.plant.Q, max(float(self.plant.R[0, 0]), EPS)
        noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(process_noise)
        B1 = noise_eigenvectors @ np.diag(np.sqrt(np.maximum(noise_eigenvalues, 0.0)))
        return A, B, C, D, C1_x, C1_u, B1, measurement_noise

    def check_stabilisability_and_detectability(self):
        """Check continuous-time stabilisability and detectability using Popov-Belevitch-Hautus (PBH) tests."""

        # get the plant matrices
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        state_count = A.shape[0]
        tolerance = EPS * max(1.0, np.linalg.norm(A, ord=2))

        # get the open-loop eigenvalues that are close to zero or positive (marginally stable or unstable)
        unstable_modes = [eigenvalue for eigenvalue in np.linalg.eigvals(A) if eigenvalue.real >= -tolerance]

        uncontrollable_modes = []
        unobservable_modes = []
        for eigenvalue in unstable_modes:
            # compute the controllability and observability matrices
            controllability_pbh = np.hstack([eigenvalue * np.eye(state_count) - A, B])
            observability_pbh = np.vstack([eigenvalue * np.eye(state_count) - A, C])
            # rank (PBH) tests for stabilisability and detectability
            if np.linalg.matrix_rank(controllability_pbh, tol=tolerance) < state_count:
                uncontrollable_modes.append(eigenvalue)
            if np.linalg.matrix_rank(observability_pbh, tol=tolerance) < state_count:
                unobservable_modes.append(eigenvalue)

        # do there exist uncontrollable or unobservable modes that are unstable?
        is_stabilisable = not uncontrollable_modes
        is_detectable = not unobservable_modes
        self.logger.debug(f'H-infinity plant stabilisability and detectability check: \n'
            f'stabilisable: {is_stabilisable}, detectable: {is_detectable}, '
            f'uncontrollable unstable modes: {uncontrollable_modes}, unobservable unstable modes: {unobservable_modes}')

        return is_stabilisable, is_detectable

    def find_care_design(self, A, B, C, C1_x, C1_u, B1, measurement_noise):
        """Solve the coupled CARE/FARE conditions and apply bisection algorithm to find the minimum gamma
        such that these equations have positive definite solutions."""

        process_covariance = B1 @ B1.T
        input_cost = C1_u ** 2
        state_cost = C1_x.T @ C1_x

        # if B1 and measurement_noise are all zero, raise a warning
        print(f'B1: {B1}, measurement_noise: {measurement_noise}')
        if np.allclose(B1, 0.0) and np.isclose(measurement_noise, 0.0):
            self.logger.warning('H-infinity design: both process noise and measurement noise are zero; '
                'the controller will be equivalent to an open-loop controller.')

        def design_at_gamma(gamma):
            try:
                # The indefinite CARE preserves rank-deficient disturbance and
                # control matrices, unlike Cholesky factorization of their difference.
                B1_B = np.hstack([B1, B])
                control_riccati_weights = block_diag(-gamma ** 2 * np.eye(B1.shape[1]), input_cost * np.eye(B.shape[1]))
                X = solve_continuous_are(A, B1_B, state_cost, control_riccati_weights)
                F = B.T @ X / input_cost

                A_hat = A + process_covariance @ X / gamma ** 2
                CF = np.hstack([C.T, F.T])
                filter_riccati_weights = block_diag(measurement_noise * np.eye(C.shape[0]), -gamma ** 2 / input_cost * np.eye(F.shape[0]))
                Y = solve_continuous_are(A_hat.T, CF, process_covariance, filter_riccati_weights)
                if np.max(np.abs(np.linalg.eigvals(X @ Y))) >= gamma ** 2:
                    return None
                H = Y @ C.T / measurement_noise
                A_controller = A_hat - B @ F - H @ C
                B_controller = -H
                C_controller = F
                D_controller = np.zeros((1, 1))
                return X, Y, F, H, A_controller, B_controller, C_controller, D_controller
            except (ValueError, np.linalg.LinAlgError):
                return None

        # apply bisection search to find the minimum feasible gamma
        gamma_high = 1.0
        design_high = design_at_gamma(gamma_high)
        while design_high is None and gamma_high < 1 / EPS:
            gamma_high *= 2.0
            design_high = design_at_gamma(gamma_high)
        if design_high is None:
            raise ValueError('CARE/FARE H-infinity design failed to find a feasible gamma; check plant stabilisability and detectability.')

        gamma_low = 0.0
        for _ in range(45):
            gamma_mid = (gamma_low + gamma_high) / 2.0
            design_mid = design_at_gamma(gamma_mid)
            if design_mid is None:
                gamma_low = gamma_mid
            else:
                gamma_high, design_high = gamma_mid, design_mid
        return gamma_high, design_high

    def update_controller_matrices(self):
        """Design and cache the controller using the CARE/FARE method."""
        A, B, C, D, C1_x, C1_u, B1, measurement_noise = self._get_design_data()
        design_matrices = (A, B, C, D, self.plant.Q, self.plant.R, C1_x)
        design_key = (C1_u, self.simulator.dt_anim, *((matrix.shape, matrix.dtype.str, matrix.tobytes()) for matrix in design_matrices))
        if design_key == self._design_key:
            return
        if not np.allclose(D, 0.0):
            raise ValueError('H-infinity synthesis currently requires D=0, matching the generalized plant used by the CARE/FARE formulas.')
        is_stabilisable, is_detectable = self.check_stabilisability_and_detectability()
        if not is_stabilisable or not is_detectable:
            raise ValueError('H-infinity synthesis requires a stabilisable and detectable plant. '
                f'Stabilisable={is_stabilisable}, detectable={is_detectable}; '
                'see the debug log for the unstable or marginal modes that failed the PBH tests.')

        gamma, design = self.find_care_design(A, B, C, C1_x, C1_u, B1, measurement_noise)
        self.X, self.Y, self.F, self.H, self.A_K, self.B_K, self.C_K, self.D_K = design
        self.gamma = gamma

        # exact sampled transition of the continuous controller for a held measured-output deviation.
        P_K_aug = np.zeros((A.shape[0] + 1, A.shape[0] + 1))
        P_K_aug[:-1, :-1] = self.A_K
        P_K_aug[:-1, -1:] = self.B_K
        P_K_step = expm(P_K_aug * self.simulator.dt_anim)
        self.A_Kd = P_K_step[:-1, :-1]
        self.B_Kd = P_K_step[:-1, -1:]
        self._design_key = design_key
        self.logger.info(f'H-infinity controller designed using CARE/FARE: gamma={gamma}, C1_x={C1_x}, C1_u={C1_u}')
        self.logger.debug(f'H-infinity controller design matrices: F={self.F}, H={self.H}, A_K={self.A_K}, B_K={self.B_K}, C_K={self.C_K}, D_K={self.D_K}')

    def is_closed_loop_stable_discrete(self) -> tuple[bool, np.ndarray]:
        """Check poles of the sampled plant/controller loop."""
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        dims = self.plant.dims
        A_aug = np.block([[A, B], [np.zeros((1, dims + 1))]])
        I_aug = np.eye(dims + 1)
        A_aug_step = I_aug.copy()
        integration_steps = np.diff(get_t_span(0.0, self.simulator.dt_anim, self.plant.dt_int))
        method = self.simulator.integrator_method
        if method in (IntegratorType.ANALYTIC_ODE, IntegratorType.ANALYTIC_SDE):
            A_aug_step = expm(A_aug * self.simulator.dt_anim)
        else:
            for step in integration_steps:
                scaled_A = A_aug * step
                if method is IntegratorType.EULER_MARUYAMA:
                    step_matrix = I_aug + scaled_A
                else:
                    A2 = scaled_A @ scaled_A
                    A3 = A2 @ scaled_A
                    step_matrix = I_aug + scaled_A + A2 / 2 + A3 / 6 + (A3 @ scaled_A) / 24
                A_aug_step = step_matrix @ A_aug_step

        A_d, B_d = A_aug_step[:dims, :dims], A_aug_step[:dims, dims:]
        current_output_gain = self.C_K @ self.B_Kd + self.D_K
        A_cl = np.block([
            [A_d + B_d @ current_output_gain @ C,
             B_d @ self.C_K @ self.A_Kd,
             B_d @ current_output_gain @ D],
            [self.B_Kd @ C, self.A_Kd, self.B_Kd @ D],
            [current_output_gain @ C,
             self.C_K @ self.A_Kd,
             current_output_gain @ D],
        ])
        poles = np.linalg.eigvals(A_cl)
        return bool(np.all(np.abs(poles) < 1.0)), poles

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        """Calculate the H-infinity control input for the current setpoint."""
        if not isinstance(e, np.ndarray) or e.shape != (1, 1):
            raise ValueError('e must have shape (1, 1)')
        self.update_controller_matrices()

        stable, poles = self.is_closed_loop_stable_discrete()
        if not stable and self.cl_stable_prev:
            self.logger.warning(f'Closed-loop unstable for current H-infinity parameters; z-plane poles: {poles}.')
        self.cl_stable_prev = stable

        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        dims = self.plant.dims
        equilibrium_matrix = np.block([[A, B], [C, D]])
        equilibrium_rhs = np.vstack([np.zeros((dims, 1)), self.simulator.y_sp])
        try:
            equilibrium = np.linalg.solve(equilibrium_matrix, equilibrium_rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError('H-infinity controller cannot track this setpoint because the plant has no unique steady-state solution.') from error
        u_ss = equilibrium[dims:]

        y_deviation = -e
        if self.x_controller.shape != (dims, 1):
            self.reset_memory()
        self.x_controller = self.A_Kd @ self.x_controller + self.B_Kd @ y_deviation
        u_deviation = self.C_K @ self.x_controller + self.D_K @ y_deviation
        return u_ss + u_deviation


class ModelPredictiveController:
    """
    A model predictive controller (MPC) that computes the optimal control input by solving a quadratic program
    over a finite prediction horizon. The MPC uses a linear model of the plant, along with user-defined cost function
    weights and constraints, to determine the control actions that minimise the predicted error while satisfying
    input and state constraints.

    The parameters of the controller are:

    - The plant model state space matrices, A_hat, B_hat, C_hat, D_hat, which may differ from the actual plant matrices.
    In the GUI, there is a button to inherit the actual plant matrices or to input their own model matrices.
    - Horizon length, N, an integer
    - Value function matrices and scalars, for each i between 0 and N-1 (select i in a dropdown in the GUI):
    -- V_xx: state cost matrix (shape: (n, n)), cost is Δx_i.T @ V_xx @ Δx_i
    -- V_uu: input cost factor (scalar), cost is V_uu_i * Δu_i ** 2
    -- V_yy: output cost factor (scalar), cost is V_yy_i * Δy_i ** 2
    -- V_xNxN: terminal state cost matrix (shape: (n, n)), cost is Δx_N.T @ V_xNxN @ Δx_N
    -- M: state constraint matrix, N: input constraint vector and b: constant constraint vector, such that
       M_i @ Δx_i + N_i @ Δu_i <= b_i - M_i @ x_ss - N_i @ u_ss, for i = 0, ..., N-1
       M_N @ Δx_N <= b_N - M_N @ x_ss

    Steps of the algorithm:

    1. Calculate the steady state values of x, u and y (setpoint) by solving the linear equations
    A_hat @ x_ss + B_hat @ u_ss = 0 and C_hat @ x_ss + D_hat @ u_ss = y_sp, and then calculate
    the state deviations Δx = x_hat_0 - x_ss, Δu = u - u_ss and Δy = y - y_sp.

    2. Given the controller's continuous-time model matrices A_hat, B_hat, C_hat, D_hat, discretise them using the same integration method
    and substeps as the simulator to get discrete-time matrices A_d_hat, B_d_hat, C_d_hat, D_d_hat, such that
      Δx_{i+1} = A_d_hat @ Δx_i + B_d_hat @ Δu_i, Δy_i = C_d_hat @ Δx_i + D_d_hat @ Δu_i.

    3. Use a Kalman filter to estimate the current state x_0 (and hence Δx_0 = x_0 - x_ss) based on the previous state estimate,
    the previous control input, and the current measurement. Take the noise covariance matrices from the plant,
    and the plant model matrices from the MPC controller's model matrices (which may differ from the actual plant matrices).

    4. Formulate the quadratic program (QP):
      min {Δx_i, Δu_i} sum_{i=0}^{N-1} [Δx_i.T @ V_xx_i @ Δx_i + V_uu_i * Δu_i ** 2 + V_yy_i * (C_d_hat @ Δx_i + D_d_hat @ Δu_i) ** 2] + Δx_N.T @ V_xx_N @ Δx_N
      such that {
        Δx_i = Δx_0,  (initial condition)
        Δx_{i+1} = A_d_hat @ Δx_i + B_d_hat @ Δu_i,  for i = 0, ..., N-1  (dynamics obey internal model of plant)
        Δy_i = C_d_hat @ Δx_i + D_d_hat @ Δu_i,  for i = 0, ..., N-1
        M_i @ Δx_i + N_i @ Δu_i <= b_i - M_i @ x_ss - N_i @ u_ss,  for i = 0, ..., N-1  (state and input constraints)
        M_N @ Δx_N <= b_N - M_N @ x_ss  (terminal state constraint)
      }
    These matrices should be used to calculate the stacked form of the QP, which is a single quadratic program with decision variables
    Δx_0, Δu_0, ..., Δx_{N-1}, Δu_{N-1}, Δx_N.

    5. Solve the QP using the OSQP solver to get the optimal control input Δu_0, calculate u = u_ss + Δu_0 and return u.
    """

    def __init__(self, simulator, plant):
        self.simulator = simulator
        self.plant = plant
        self.logger = get_logger()
        self.A_hat = plant.A.copy()
        self.B_hat = plant.B.copy()
        self.C_hat = plant.C.copy()
        self.D_hat = plant.D.copy()
        self._discrete_model_key = None

        # Copy each default matrix so edits at one stage cannot affect another.
        initial_V_xx = np.asarray(GUI_SLIDER_CONFIG['MPC_V_xx']['init'], dtype=float)
        if initial_V_xx.shape != (plant.dims, plant.dims):
            initial_V_xx = np.eye(plant.dims)
        self._initial_V_xx = initial_V_xx.copy()
        self._initial_V_yy = float(GUI_SLIDER_CONFIG['MPC_V_yy']['init'])
        self._initial_V_uu = float(GUI_SLIDER_CONFIG['MPC_V_uu']['init'])
        self.costs = {}
        self.constraints = {}
        self._configured_horizon = None
        self.ensure_horizon(int(getattr(simulator, 'MPC_N', 5)))
        self.reset_memory()

    def _default_cost(self):
        return {
            'V_xx': self._initial_V_xx.copy(),
            'V_yy': self._initial_V_yy,
            'V_uu': self._initial_V_uu,
        }

    @staticmethod
    def _default_constraint(state_count):
        M = np.asarray(GUI_SLIDER_CONFIG['MPC_M']['init'], dtype=float)
        N = np.asarray(GUI_SLIDER_CONFIG['MPC_N_i']['init'], dtype=float)
        b = np.asarray(GUI_SLIDER_CONFIG['MPC_b']['init'], dtype=float)
        if M.ndim != 2 or M.shape[1] != state_count or M.shape[0] != 0:
            M = np.empty((0, state_count))
        if N.size == 0:
            N = np.empty((0, 1))
        if b.size == 0:
            b = np.empty((0, 1))
        return {
            'M': M.copy(),
            'N': N.copy(),
            'b': b.copy(),
        }

    def ensure_horizon(self, horizon):
        """Resize stage settings, carrying the old terminal forward on expansion."""
        horizon = int(horizon)
        if not 1 <= horizon <= 20:
            raise ValueError('MPC horizon must be an integer between 1 and 20.')

        previous_horizon = self._configured_horizon
        if previous_horizon is not None and horizon > previous_horizon:
            previous_running_cost = copy.deepcopy(self.costs[previous_horizon - 1])
            previous_terminal_cost = copy.deepcopy(self.costs[previous_horizon])
            previous_running_constraint = copy.deepcopy(self.constraints[previous_horizon - 1])
            previous_terminal_constraint = copy.deepcopy(self.constraints[previous_horizon])

            # The old terminal becomes a running stage: fill every newly
            # exposed running stage with the previous running-stage settings.
            for stage in range(previous_horizon, horizon):
                self.costs[stage] = copy.deepcopy(previous_running_cost)
                self.constraints[stage] = copy.deepcopy(previous_running_constraint)

            # Keep terminal-only state cost and constraints at the new endpoint.
            self.costs[horizon] = previous_terminal_cost
            self.constraints[horizon] = previous_terminal_constraint
        else:
            for stage in range(horizon + 1):
                self.costs.setdefault(stage, self._default_cost())
                self.constraints.setdefault(stage, self._default_constraint(self.plant.dims))

        self._configured_horizon = horizon

    def reset_for_state_dimension(self):
        """Reset model-dependent MPC settings after the true plant dimension changes."""
        self.set_model_matrices(self.plant.A, self.plant.B, self.plant.C, self.plant.D)
        self._initial_V_xx = np.eye(self.plant.dims)
        self.costs.clear()
        self.constraints.clear()
        self._configured_horizon = None
        self.ensure_horizon(int(getattr(self.simulator, 'MPC_N', 5)))

    def set_model_matrices(self, A, B, C, D):
        """Set the internal model, whose state dimension must match the true plant."""
        matrices = tuple(np.asarray(matrix, dtype=float) for matrix in (A, B, C, D))
        A, B, C, D = matrices
        state_count = self.plant.dims
        expected_shapes = ((state_count, state_count), (state_count, 1), (1, state_count), (1, 1),)
        if any(matrix.shape != shape or not np.all(np.isfinite(matrix)) for matrix, shape in zip(matrices, expected_shapes)):
            raise ValueError('MPC model matrices must be finite and have the same state dimension as the true plant.')

        # Replace all four matrices together so the model cannot be left partially updated.
        self.A_hat, self.B_hat, self.C_hat, self.D_hat = (matrix.copy() for matrix in matrices)
        self._discrete_model_key = None
        self.observer = None

    def reset_memory(self):
        """Reset the state observer to the configured initial state."""
        self.observer = None
        if not hasattr(self, 'A_d_hat'):
            return
        if self.A_d_hat.shape != self.A_hat.shape:
            return
        self.observer = KalmanFilter(self.A_d_hat, self.B_d_hat, self.C_d_hat, self.D_d_hat, self.Q_d_hat, self.R_d,
                                    initial_state=self.plant.x_0.copy())

    def update_discrete_model(self):
        """Stage 2: discretise the internal model using the simulator's substeps."""
        integration_method = self.simulator.integrator_method
        use_ode_mode = self.simulator.use_ode_mode
        if use_ode_mode and integration_method not in (IntegratorType.RK4, IntegratorType.ANALYTIC_ODE):
            raise ValueError(f'Invalid integration method for ODE mode: {integration_method}.')
        if not use_ode_mode and integration_method not in (IntegratorType.EULER_MARUYAMA, IntegratorType.ANALYTIC_SDE):
            raise ValueError(f'Invalid integration method for SDE mode: {integration_method}.')

        integration_steps = np.diff(get_t_span(0.0, self.simulator.dt_anim, self.simulator.dt_int))
        model_matrices = (self.A_hat, self.B_hat, self.C_hat, self.D_hat)
        model_key = (integration_method, use_ode_mode, self.simulator.dt_anim, self.simulator.dt_int,
            *((matrix.shape, matrix.dtype.str, matrix.tobytes()) for matrix in model_matrices),
            self.plant.Q.shape, self.plant.Q.dtype.str, self.plant.Q.tobytes(), self.plant.R.tobytes())
        if model_key == self._discrete_model_key:
            return

        # calculate the discrete-time model matrices
        self.A_d_hat, self.B_d_hat = discretise_state_space(self.A_hat, self.B_hat, integration_steps, integration_method)
        self.C_d_hat = self.C_hat.copy()
        self.D_d_hat = self.D_hat.copy()
        self.Q_d_hat = discretise_process_noise(self.A_hat, self.plant.Q, integration_steps, integration_method, use_ode_mode)
        self.R_d = np.array([[max(float(self.plant.R[0, 0]), EPS)]])
        self._discrete_model_key = model_key
        self.reset_memory()

    def get_steady_state(self):
        """Stage 1: find the internal model equilibrium at the current setpoint."""
        state_count = self.A_hat.shape[0]
        equilibrium_matrix = np.block([[self.A_hat, self.B_hat], [self.C_hat, self.D_hat]])
        equilibrium_rhs = np.vstack([np.zeros((state_count, 1)), self.simulator.y_sp])
        try:
            equilibrium = np.linalg.solve(equilibrium_matrix, equilibrium_rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError('MPC cannot track the setpoint because the internal model has no unique steady-state solution.') from error
        return equilibrium[:state_count], equilibrium[state_count:]

    @staticmethod
    def _validate_state_cost(cost, state_count, stage):
        cost = np.asarray(cost, dtype=float)
        if (cost.shape != (state_count, state_count) or not np.all(np.isfinite(cost)) or not np.allclose(cost, cost.T)
                or np.min(np.linalg.eigvalsh(cost)) < -EPS):
            raise ValueError(f'MPC V_xx at stage {stage} must be a finite symmetric positive-semidefinite {state_count}x{state_count} matrix.')
        return 0.5 * (cost + cost.T)

    @staticmethod
    def _validate_scalar_cost(cost, name, stage):
        cost = float(cost)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(f'MPC {name} at stage {stage} must be finite and nonnegative.')
        return cost

    @staticmethod
    def _validate_constraint(constraint, state_count, stage, terminal=False):
        M_i = np.asarray(constraint['M'], dtype=float)
        N_i = np.asarray(constraint['N'], dtype=float)
        b_i = np.asarray(constraint['b'], dtype=float)
        if M_i.ndim != 2 or M_i.shape[1] != state_count:
            raise ValueError(f'MPC M at stage {stage} must have {state_count} columns.')
        if b_i.shape != (M_i.shape[0], 1):
            raise ValueError(f'MPC b at stage {stage} must have one entry per row of M.')
        if not terminal and N_i.shape != (M_i.shape[0], 1):
            raise ValueError(f'MPC N at stage {stage} must have one entry per row of M.')
        matrices_to_check = (M_i, b_i) if terminal else (M_i, N_i, b_i)
        if not all(np.all(np.isfinite(matrix)) for matrix in matrices_to_check):
            raise ValueError(f'MPC constraints at stage {stage} must be finite.')
        return M_i, N_i, b_i

    def _assemble_qp(self, initial_state_deviation, x_ss, u_ss, horizon):
        """Stage 4: build the stacked quadratic objective and linear constraints."""
        state_count = self.A_d_hat.shape[0]
        block_size = state_count + 1
        terminal_offset = horizon * block_size
        decision_count = terminal_offset + state_count
        P = np.zeros((decision_count, decision_count))
        q = np.zeros(decision_count)
        equality_rows = []
        equality_values = []
        inequality_rows = []
        inequality_values = []

        def state_indices(stage):
            start = stage * block_size
            return slice(start, start + state_count)

        def input_index(stage):
            return stage * block_size + state_count

        for stage in range(horizon):
            cost = self.costs[stage]
            V_xx = self._validate_state_cost(cost['V_xx'], state_count, stage)
            V_yy = self._validate_scalar_cost(cost['V_yy'], 'V_yy', stage)
            V_uu = self._validate_scalar_cost(cost['V_uu'], 'V_uu', stage)
            x_ind = np.arange(state_indices(stage).start, state_indices(stage).stop)
            u_ind = input_index(stage)

            # The output penalty is V_yy * (C_d @ Δx + D_d @ Δu)^2.
            output_vector = np.hstack([self.C_d_hat, self.D_d_hat]).reshape(-1)
            local_hessian = np.zeros((block_size, block_size))
            local_hessian[:state_count, :state_count] = V_xx
            local_hessian += V_yy * np.outer(output_vector, output_vector)
            local_hessian[-1, -1] += V_uu
            stage_indices = np.hstack([x_ind, [u_ind]])
            # OSQP's objective has a leading 1/2, hence multiply each
            # quadratic stage-cost matrix by two for the equivalent Hessian.
            P[np.ix_(stage_indices, stage_indices)] += 2.0 * local_hessian

            if stage == 0:
                for state_index in range(state_count):
                    row = np.zeros(decision_count)
                    row[state_indices(0).start + state_index] = 1.0
                    equality_rows.append(row)
                    equality_values.append(float(initial_state_deviation[state_index, 0]))

            next_state_start = state_indices(stage + 1).start if stage + 1 < horizon else terminal_offset
            for state_index in range(state_count):
                row = np.zeros(decision_count)
                row[next_state_start + state_index] = 1.0
                row[state_indices(stage)] -= self.A_d_hat[state_index]
                row[input_index(stage)] -= self.B_d_hat[state_index, 0]
                equality_rows.append(row)
                equality_values.append(0.0)

            M_i, N_i, b_i = self._validate_constraint(self.constraints[stage], state_count, stage)
            for constraint_index in range(M_i.shape[0]):
                row = np.zeros(decision_count)
                row[state_indices(stage)] = M_i[constraint_index]
                row[input_index(stage)] = N_i[constraint_index, 0]
                bound = b_i[constraint_index, 0] - (M_i[constraint_index:constraint_index + 1] @ x_ss)[0, 0] - N_i[constraint_index, 0] * float(u_ss[0, 0])
                inequality_rows.append(row)
                inequality_values.append(float(bound))

        # Terminal cost and constraint only apply to Δx_N.
        terminal_cost = self.costs[horizon]
        terminal_V_xx = self._validate_state_cost(terminal_cost['V_xx'], state_count, horizon)
        P[terminal_offset:, terminal_offset:] += 2.0 * terminal_V_xx
        M_N, _, b_N = self._validate_constraint(self.constraints[horizon], state_count, horizon, terminal=True)
        for constraint_index in range(M_N.shape[0]):
            row = np.zeros(decision_count)
            row[terminal_offset:] = M_N[constraint_index]
            bound = b_N[constraint_index, 0] - (M_N[constraint_index:constraint_index + 1] @ x_ss)[0, 0]
            inequality_rows.append(row)
            inequality_values.append(float(bound))

        # Initial condition and dynamics are equalities; user constraints are
        # upper-bounded inequalities. Stack both for OSQP's l <= A z <= u form.
        constraint_rows = equality_rows + inequality_rows
        lower = equality_values + [-np.inf] * len(inequality_rows)
        upper = equality_values + inequality_values
        constraint_matrix = sparse.csc_matrix(np.vstack(constraint_rows))
        return (sparse.triu(sparse.csc_matrix(P), format='csc'), q, constraint_matrix,
            np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), input_index(0))

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        """Execute the five MPC stages and return the first optimized input."""
        if not isinstance(e, np.ndarray) or e.shape != (1, 1):
            raise ValueError('e must have shape (1, 1)')

        # step 1: calculate the equilibrium and prepare state deviations.
        x_ss, u_ss = self.get_steady_state()
        horizon = int(getattr(self.simulator, 'MPC_N', 5))
        self.ensure_horizon(horizon)

        # step 2: reuse the shared state-space discretization for this frame.
        self.update_discrete_model()

        # step 3: estimate the state from the previous input and current reading.
        y_meas = self.simulator.y_sp - e
        x_hat_0 = self.observer.update(y_meas, self.plant.u)
        delta_x_0 = x_hat_0 - x_ss

        # step 4: assemble the single sparse quadratic program.
        P, q, A_qp, lower, upper, delta_u_0_index = self._assemble_qp(
            delta_x_0, x_ss, u_ss, horizon
        )

        # step 5: solve the QP, then apply only its first control move.
        import osqp

        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A_qp, l=lower, u=upper,
            verbose=False, eps_abs=EPS, eps_rel=EPS, max_iter=20000, polishing=True)
        result = solver.solve()
        if result.info.status_val not in (1, 2) or result.x is None:
            self.logger.error(f'MPC QP failed: status={result.info.status}, status_val={result.info.status_val}')
            raise RuntimeError(f'MPC quadratic program failed: {result.info.status}.')
        return u_ss + np.array([[result.x[delta_u_0_index]]])


class KalmanFilter:
    """Discrete-time Kalman filter implemented from first principles."""

    def __init__(self, A, B, C, D, Q, R, initial_state=None, initial_covariance=None):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        state_count = self.A.shape[0]
        self._initial_state = np.zeros((state_count, 1)) if initial_state is None else np.asarray(initial_state, dtype=float).reshape(state_count, 1)
        self._initial_covariance = np.eye(state_count) if initial_covariance is None else np.asarray(initial_covariance, dtype=float)
        self.reset()

    def reset(self):
        """Reset state estimate and uncertainty to their initial values."""
        self.x_hat = self._initial_state.copy()
        self.P = self._initial_covariance.copy()
        self._has_measurement = False

    def update(self, measurement, previous_input):
        """Predict with the previous input and correct using the current measurement."""
        measurement = np.asarray(measurement, dtype=float).reshape(self.C.shape[0], 1)
        previous_input = np.asarray(previous_input, dtype=float).reshape(self.B.shape[1], 1)

        # Prediction is omitted on the first call because the initial state is
        # already at the timestamp of the first measurement.
        if self._has_measurement:
            self.x_hat = self.A @ self.x_hat + self.B @ previous_input
            self.P = self.A @ self.P @ self.A.T + self.Q
        else:
            self._has_measurement = True

        innovation = measurement - self.C @ self.x_hat - self.D @ previous_input
        innovation_covariance = self.C @ self.P @ self.C.T + self.R
        kalman_gain = np.linalg.solve(innovation_covariance, self.C @ self.P).T
        self.x_hat = self.x_hat + kalman_gain @ innovation

        # Joseph-form covariance update maintains symmetry and non-negativity.
        identity_minus_gain_C = np.eye(self.A.shape[0]) - kalman_gain @ self.C
        self.P = identity_minus_gain_C @ self.P @ identity_minus_gain_C.T + kalman_gain @ self.R @ kalman_gain.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x_hat.copy()
