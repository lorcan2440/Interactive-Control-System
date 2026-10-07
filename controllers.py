# built-ins
from enum import Enum, auto

# external imports
import numpy as np
from scipy.linalg import block_diag, expm
from scipy.linalg import solve_continuous_are

# local imports
from integrators import IntegratorType
from utils import get_logger, EPS, get_t_span


class ControllerType(Enum):

    NONE = auto()
    MANUAL = auto()
    OPENLOOP = auto()
    BANGBANG = auto()
    PID = auto()
    H2 = auto()
    HINF = auto()
    
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
        
        In principle, we could apply an impulse input for one frame, using the formula:
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
        and previous measurement and derivative action used in filtered derivative.
        '''
        self.e_integrated = np.array([[0.0]])
        self.y_meas_prev = np.array([[0.0]])
        self.u_d_prev = np.array([[0.0]])
        self.cl_stable_prev = True

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        """
        Compute PID control input using P and I on the error, and a low-pass filtered derivative on the measurement.
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

        # get measurement
        y_meas = self.sim.y_sp - e

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

        # filtered derivative on measurement
        if self.K_d == 0:
            u_d = np.array([[0.0]])
        else:
            # low-pass filter time constant: use user-configured `tau` when available,
            # otherwise fall back to 5x the sampling period
            # NOTE: consider setting this to 0.1x the derivative time constant K_p / K_d
            tau = getattr(self, 'tau', max(5.0 * dt, 1e-6))

            # Tustin's method implementation of the first-order low-pass 
            # filter on the derivative term, with input y and output u_d
            alpha = 1.0 - dt / tau
            alpha = max(min(alpha, 1.0), 0.0)

            u_d = alpha * self.u_d_prev - (self.K_d / tau) * (y_meas - self.y_meas_prev)
            self.u_d_prev = u_d

        # update stored noisy measurement
        self.y_meas_prev = y_meas

        # total control input = P + I + D
        u = u_p + u_i + u_d

        return u
    
    def K_y(self, s: complex) -> complex:
        '''
        Controller transfer function (continuous-time) from y_meas to u.
        '''
        return self.K_p * (self.T_d * s / (self.tau * s + 1) - 1 - 1 / (self.T_i * s))
    
    def K_sp(self, s: complex) -> complex:
        '''
        Controller transfer function (continuous-time) from y_sp to u.
        '''
        return self.K_p * (1 + 1 / (self.T_i * s))
    
    def K(self, s: complex) -> np.ndarray:

        '''
        Controller transfer function (continuous-time) from [y_meas, y_sp]^T to u, 
        suitable for use in the generalised plant and controller interconnection.

        The first entry is the TF from y_meas to u. The second entry is the TF from y_sp to u.

        This is the form for the PID controller with filtered derivative on the measurement.
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
        if tau <= 0:
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
        alpha = max(min(1.0 - dt_anim / tau, 1.0), 0.0)
        A_Kd = np.zeros((3, 3))
        B_Kd = np.array([[0.0], [1.0], [0.0]])
        if K_i != 0:
            A_Kd[0, 0] = 1.0
            B_Kd[0, 0] = -dt_anim
        if K_d != 0:
            A_Kd[2, 1:] = [K_d / tau, alpha]
            B_Kd[2, 0] = -K_d / tau
        C_Kd = np.array([[K_i, K_d / tau, alpha]])
        D_Kd = np.array([[-K_p - K_i * dt_anim - K_d / tau,]])

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
        C1_x = np.array([float(getattr(self.simulator, f'C1_x{i + 1}', 1.0)) for i in range(dims)]).reshape(1, dims)
        C1_u = np.array([[float(getattr(self.simulator, 'C1_u', 1.0))]])

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
        except (ValueError, np.linalg.LinAlgError) as error:
            self.logger.exception('Unable to solve H2/LQG Riccati equations.')
            raise ValueError(
                'H2 controller design failed; check that the plant is stabilisable '
                'and detectable and that its noise matrices are valid.'
            ) from error

        self.H = Y @ C.T / R_item

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
        self.logger.debug(f'H2 design updated: C1_x={C1_x}, C1_u={C1_u}, F={self.F}, H={self.H}, observer poles z={np.linalg.eigvals(A_K)}')

    def is_closed_loop_stable_discrete(self) -> tuple[bool, np.ndarray]:
        """Check stability of the sampled plant and H2 controller interconnection."""
        A, B = self.plant.A, self.plant.B
        C, D = self.plant.C, self.plant.D
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
            self.logger.warning(
                'Closed-loop unstable for current H2 parameters: '
                'eigenvalues of A_cl (poles in z-plane) are %s.',
                cl_z_poles,
            )
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
        C1_x = np.array([
            float(getattr(self.simulator, f'Hinf_C1_x{i + 1}', 1.0))
            for i in range(dims)
        ]).reshape(1, dims)
        C1_u = float(getattr(self.simulator, 'Hinf_C1_u', 1.0))
        if C1_u <= 0.0:
            raise ValueError('H-infinity input weight Hinf_C1_u must be greater than zero.')

        process_noise, measurement_noise = self.plant.Q, max(float(self.plant.R[0, 0]), EPS)
        noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(process_noise)
        B1 = noise_eigenvectors @ np.diag(np.sqrt(np.maximum(noise_eigenvalues, 0.0)))
        return A, B, C, D, C1_x, C1_u, B1, measurement_noise

    def check_stabilisability_and_detectability(self):
        """Check continuous-time stabilisability and detectability using PBH tests."""
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        state_count = A.shape[0]
        tolerance = 1e-9 * max(1.0, np.linalg.norm(A, ord=2))
        unstable_modes = [
            eigenvalue for eigenvalue in np.linalg.eigvals(A)
            if eigenvalue.real >= -tolerance
        ]

        uncontrollable_modes = []
        unobservable_modes = []
        for eigenvalue in unstable_modes:
            controllability_pbh = np.hstack([
                eigenvalue * np.eye(state_count) - A, B
            ])
            observability_pbh = np.vstack([
                eigenvalue * np.eye(state_count) - A, C
            ])
            if np.linalg.matrix_rank(controllability_pbh, tol=tolerance) < state_count:
                uncontrollable_modes.append(eigenvalue)
            if np.linalg.matrix_rank(observability_pbh, tol=tolerance) < state_count:
                unobservable_modes.append(eigenvalue)

        is_stabilizable = not uncontrollable_modes
        is_detectable = not unobservable_modes
        self.logger.debug(
            'H-infinity plant tests: stabilizable=%s, detectable=%s, '
            'uncontrollable unstable modes=%s, unobservable unstable modes=%s',
            is_stabilizable,
            is_detectable,
            uncontrollable_modes,
            unobservable_modes,
        )
        return is_stabilizable, is_detectable

    def find_care_design(self, A, B, C, C1_x, C1_u, B1, measurement_noise):
        """Solve the coupled CARE/FARE conditions and bisect for minimum gamma."""
        process_covariance = B1 @ B1.T
        input_cost = C1_u ** 2
        state_cost = C1_x.T @ C1_x

        def design_at_gamma(gamma):
            try:
                # The indefinite CARE preserves rank-deficient disturbance and
                # control matrices, unlike Cholesky factorization of their difference.
                control_riccati_inputs = np.hstack([B1, B])
                control_riccati_weights = block_diag(
                    -gamma ** 2 * np.eye(B1.shape[1]),
                    input_cost * np.eye(B.shape[1]),
                )
                X = solve_continuous_are(
                    A,
                    control_riccati_inputs,
                    state_cost,
                    control_riccati_weights,
                )
                F = B.T @ X / input_cost

                A_hat = A + process_covariance @ X / gamma ** 2
                filter_riccati_inputs = np.hstack([C.T, F.T])
                filter_riccati_weights = block_diag(
                    measurement_noise * np.eye(C.shape[0]),
                    -gamma ** 2 / input_cost * np.eye(F.shape[0]),
                )
                Y = solve_continuous_are(
                    A_hat.T,
                    filter_riccati_inputs,
                    process_covariance,
                    filter_riccati_weights,
                )
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

        gamma_high = 1.0
        design_high = design_at_gamma(gamma_high)
        while design_high is None and gamma_high < 1e8:
            gamma_high *= 2.0
            design_high = design_at_gamma(gamma_high)
        if design_high is None:
            raise ValueError(
                'CARE/FARE H-infinity design failed to find a feasible gamma; '
                'check plant stabilisability and detectability.'
            )

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
        design_key = (
            C1_u,
            self.simulator.dt_anim,
            *((matrix.shape, matrix.dtype.str, matrix.tobytes()) for matrix in design_matrices),
        )
        if design_key == self._design_key:
            return
        if not np.allclose(D, 0.0):
            raise ValueError(
                'H-infinity synthesis currently requires D=0, matching the generalized plant '
                'used by the CARE/FARE formulas.'
            )
        is_stabilisable, is_detectable = self.check_stabilisability_and_detectability()
        if not is_stabilisable or not is_detectable:
            raise ValueError(
                'H-infinity synthesis requires a stabilisable and detectable plant. '
                f'Stabilisable={is_stabilisable}, detectable={is_detectable}; '
                'see the debug log for the unstable or marginal modes that failed the PBH tests.'
            )

        gamma, design = self.find_care_design(
            A, B, C, C1_x, C1_u, B1, measurement_noise
        )

        (
            self.X, self.Y, self.F, self.H,
            self.A_K, self.B_K, self.C_K, self.D_K,
        ) = design
        self.gamma = gamma

        # Exact sampled transition of the continuous controller for a held
        # measured-output deviation.
        controller_augmented = np.zeros((A.shape[0] + 1, A.shape[0] + 1))
        controller_augmented[:-1, :-1] = self.A_K
        controller_augmented[:-1, -1:] = self.B_K
        controller_step = expm(controller_augmented * self.simulator.dt_anim)
        self.A_Kd = controller_step[:-1, :-1]
        self.B_Kd = controller_step[:-1, -1:]
        self._design_key = design_key
        self.logger.info(
            'H-infinity controller designed using CARE/FARE: gamma=%g, C1_x=%s, C1_u=%g',
            gamma,
            C1_x,
            C1_u,
        )
        self.logger.debug(
            'H-infinity controller matrices: F=%s, H=%s, A_K=%s, B_K=%s, C_K=%s, D_K=%s',
            self.F,
            self.H,
            self.A_K,
            self.B_K,
            self.C_K,
            self.D_K,
        )

    def is_closed_loop_stable_discrete(self) -> tuple[bool, np.ndarray]:
        """Check poles of the sampled plant/controller loop."""
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        dims = self.plant.dims
        A_aug = np.block([[A, B], [np.zeros((1, dims + 1))]])
        I_aug = np.eye(dims + 1)
        A_aug_step = I_aug.copy()
        integration_steps = np.diff(get_t_span(
            0.0, self.simulator.dt_anim, self.plant.dt_int
        ))
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
            self.logger.warning(
                'Closed-loop unstable for current H-infinity parameters; z-plane poles: %s',
                poles,
            )
        self.cl_stable_prev = stable

        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        dims = self.plant.dims
        equilibrium_matrix = np.block([[A, B], [C, D]])
        equilibrium_rhs = np.vstack([np.zeros((dims, 1)), self.simulator.y_sp])
        try:
            equilibrium = np.linalg.solve(equilibrium_matrix, equilibrium_rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                'H-infinity controller cannot track this setpoint because the plant has no '
                'unique steady-state solution.'
            ) from error
        u_ss = equilibrium[dims:]

        y_deviation = -e
        if self.x_controller.shape != (dims, 1):
            self.reset_memory()
        self.x_controller = self.A_Kd @ self.x_controller + self.B_Kd @ y_deviation
        u_deviation = self.C_K @ self.x_controller + self.D_K @ y_deviation
        return u_ss + u_deviation


# TODO: implement a model predictive controller (MPC), using OSQP to solve the quadratic program
# allow the user to change the model matrices (may differ from actual plant), cost function weights and horizon length
