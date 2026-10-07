# built-ins
from enum import Enum, auto

# external imports
import numpy as np
from scipy.linalg import expm

# local imports
from utils import get_logger


class ControllerType(Enum):

    NONE = auto()
    MANUAL = auto()
    OPENLOOP = auto()
    BANGBANG = auto()
    PID = auto()
    
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
        """
        cl_stable, cl_s_poles = self.is_closed_loop_stable()
        if not cl_stable and self.cl_stable_prev:  # only log one warning
            self.logger.warning(f'''Closed-loop unstable for current PID parameters: 
                eigenvalues of A_cl (poles in s-plane) are {cl_s_poles}. Prev: {self.cl_stable_prev}''')
        elif cl_stable:
            self.cl_stable_prev = True
        else:
            self.cl_stable_prev = False
        """

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
    
    def is_closed_loop_stable_continuous(self) -> tuple[bool, np.ndarray]:
        """
        Return whether the closed loop is asymptotically stable, based on the continuous-time 
        closed-loop state transition matrix A_cl.

        This function checks if the eigenvalues of the closed-loop state transition 
        matrix A_cl must have Re s < 0. The matrix A_cl is obtained by:

        - finding the state space realisation of the controller K(s) in continuous time
        - augmenting the plant state space with the controller state space
        - computing the closed-loop state transition matrix A_cl for the augmented system
        """

        # get plant matrices (continuous time)
        A = self.plant.A  # shape: (dims, dims)
        B = self.plant.B  # shape: (dims, 1)
        C = self.plant.C  # shape: (1, dims)
        D = self.plant.D  # shape: (1, 1)

        # get controller parameters
        K_p = self.K_p
        K_i = self.K_i
        K_d = self.K_d
        tau = self.tau

        # controller: K(s) = K_p + K_i / s + K_d * s / (tau * s + 1)
        # state space realisation of the PID controller (in controllable canonical form):
        A_K = np.array([[0, 1], [0, -1 / tau]])  # shape: (2, 2)
        B_K = np.array([[0], [1]])  # shape: (2, 1)
        C_K = np.array([[-K_i / tau, -(K_i + K_d / tau ** 2)]])  # shape: (1, 2)
        D_K = np.array([[K_d / tau - K_p]])  # shape: (1, 1)

        # check for ill-posed algebraic loop (D_K @ D = 1)
        M = 1 - D_K @ D
        if np.isclose(M, 0.0):
            raise ValueError(f'Ill-posed algebraic loop: 1 - D_K @ D = {M} ≈ 0.')

        # closed-loop state transition matrix
        A_cl = np.block([
            [A + B @ C * (D_K / M),  B @ C_K / M],
            [B_K @ C / M,            A_K + (D / M) * B_K @ C_K]
        ])

        # check eigenvalues of A_cl
        eigvals = np.linalg.eigvals(A_cl)

        return np.all(np.real(eigvals) < 0), eigvals
    
    def is_closed_loop_stable_discrete(self) -> tuple[bool, np.ndarray]:
        """
        Return whether the closed loop is asymptotically stable, based on the discrete-time 
        closed-loop state transition matrix A_d_cl.

        This function checks if the eigenvalues of the discretised closed-loop state transition 
        matrix A_d_cl must have |z| < 1.
        """

        # NOTE: this works, but only for proportional control with D = 0
        # TODO: generalise to allowing D != 0
        # TODO: generalise to K_i, K_d != 0, including the effect of tau

        # get plant matrices
        C = self.plant.C  # shape: (1, dims)

        # get discrete-time matrices
        A_d, B_d = self.plant.A_d, self.plant.B_d  # shapes: (dims, dims), (dims, 1)

        # when the loop is closed: u_k = K_p * (y_sp - C x_k), so
        # x_{k+1} = A_cl x_k + B_d @ K_p * y_sp, where A_cl = A_d - B_d @ K_p @ C
        A_cl = A_d - B_d @ (self.sim.K_p * C)  # closed-loop state transition matrix

        # all discrete-time poles z must have |z| < 1
        z_poles = np.linalg.eigvals(A_cl)
        return np.all(np.abs(z_poles) < 1.0), z_poles

# TODO: implement the H2 optimal controller from first principles - 
# do not just copy the below blindly as it gave suspicious results previously
# allow the user to choose the performance output z = C1 @ x + C2 @ u (user sets weight matrices C1 and C2)
# also add a function to compute the optimal H2 norm

# TODO: implement the H-infinity optimal controller from first principles -
# choose to use either the Riccati equation approach or the linear matrix inequality optimisation (solving with CVX),
# could add a GUI setting to choose which approach to use
# allow the user to choose the performance output z = C1 @ x + C2 @ u (user sets weight matrices C1 and C2)
# also add a function to compute the optimal H-infinity norm

# TODO: implement a model predictive controller (MPC), using OSQP to solve the quadratic program
# allow the user to change the model matrices (may differ from actual plant), cost function weights and horizon length

"""
# NOTE: this still uses the old interface - need to update - leave out for now

class H2Controller:

    # TODO: investigate H2 controller stability - why does the controller diverge for large C1_1?

    def __init__(self, simulator, plant):
        '''
        A H2 controller, also known as an LQG controller, is a type of optimal controller. 
        It aims to minimise the total signal energy gain of an input disturbance w 
        to the performance output signal z. The performance output is given by:

        `z = [C1 @ x, u].T`

        where `C1` is the performance gain vector, `x` is the plant state and `u` is the control input.

        In this simulation, since `x` has 2 variables, `C1` is a vector of two values: `C1_1` and `C1_2`, 
        which are the free parameters for this controler.

        The quantity being minimised is the H2 norm of the lower linear fractional transformation (LFT) of 
        the generalised plant:

        - The 'generalised plant' is a remodelled form of the plant where the inputs are the control input u 
        and disturbances w, and the outputs are the measured output y and the performance output z.
        - The 'lower LFT' T(jω) is the transfer function (TF) from w to z in the generalised plant.
        - The 'H2 norm' of a TF can be defined in either the (1) frequency or (2) time domains,

        1. ||T(s)||_2 = sqrt{integral from -∞ to ∞: T(jω)* T(jω) dω }
        2. sqrt{1/(2 pi) * integral from 0 to ∞: z(t)* z(t) dt }

        (where z(t) is the performance output to an impulse disturbance) which are equivalent due to 
        Parseval's theorem of energy conservation.

        - A larger C1_1 tends to promote minimising the effect of disturbances on x_1.
        - A larger C1_2 tends to promote minimising the effect of disturbances on x_2 (and hence y).
        - If C1_1 and C1_2 are both small, this promotes minimising the control input energy ||u||_2.
        '''

        self.simulator = simulator
        self.plant = plant
        self.h2_gains_computed = False
        self.last_C1 = None
        self.x_k = np.array([[0.0], [0.0]])  # [x1_hat, x2_hat].T

    def reset_memory(self):
        self.x_k = 0.0
        self.prev_x_ss = 0.0
        self.prev_u_ss = 0.0

    def check_for_observability(self, A: np.ndarray, C: np.ndarray) -> bool:
        '''
        Check that the pair (A, C1) is observable, required for the ARE solution and controller stability.
        '''

        # compute observability Gramian
        W_o = solve_continuous_lyapunov(A.T, -C.T @ C)

        # check if singular
        return (np.linalg.matrix_rank(W_o) == W_o.shape[0] and W_o.shape[0] == W_o.shape[1])

    def calc_u(self, e: float) -> float:
        '''
        Calculates the control input for a H2 optimal controller (aka LQG controller).
        
        ### Arguments
        - `e` (float): the error, given by y_setpoint - y_measured.
        
        ### Returns
        - `float`: the control input.
        '''

        # store as 1-element arrays
        y_measured = np.array([[self.simulator.setpoint - e]])
        e = np.array([[e]])

        # check if we need to recompute performance output vector (C1 may have changed)
        current_C1 = (self.simulator.C1_1, self.simulator.C1_2)
        
        if not self.h2_gains_computed or self.last_C1 != current_C1:
            
            # set up state space matrices
            A = np.array([[-self.plant.k12 - self.plant.d, self.plant.k21], 
                          [self.plant.k12, -self.plant.k21 - self.plant.d]])
            B1 = np.array([[0], [1]])
            B2 = np.array([[1], [0]])
            C2 = np.array([[0, 1]])

            # performance output matrix
            C1 = np.array([[self.simulator.C1_1, self.simulator.C1_2]])

            # solve algebraic Riccati equations (AREs)
            X = solve_continuous_are(A, B2, C1.T @ C1, np.eye(1))  # CARE (control ARE)
            Y = solve_continuous_are(A.T, C2.T, B1 @ B1.T, np.eye(1))  # FARE (filter ARE)

            # controller gains
            self.F = B2.T @ X    # state feedback gain
            self.H = Y @ C2.T    # Kalman gain
            self.A_cl = A + B2 @ self.F - self.H @ C2  # closed-loop observer matrix
            
            # cache system matrices for steady-state calculation
            self.A_matrix = A
            self.B2_matrix = B2
            self.C2_matrix = C2
            
            self.h2_gains_computed = True
            self.last_C1 = current_C1  # cache performance vector

            # optimal H2 norm - NOTE: MATLAB omits the sqrt(2 * pi) factor
            self.h2_norm = np.sqrt(2 * np.pi * (np.trace(B1.T @ X @ B1) + np.trace(self.F @ Y @ self.F.T)))

            # check for stability based on C1 variation
            if not self.check_for_observability(A, C1):
                warnings.warn(f'UNSTABLE at C1 = {self.C1}: observability Gramian is singular.', RuntimeWarning)

        # recalculate steady-state values at each call (setpoint may have changed)
        # compute the steady-state control input and state for the current setpoint
        # we want C2 @ x_ss = setpoint and A @ x_ss + B2 @ u_ss = 0
        M = np.vstack([
            np.hstack([self.A_matrix, self.B2_matrix]),
            np.hstack([self.C2_matrix, np.zeros((1, 1))])
        ])
        rhs = np.vstack([np.zeros((2, 1)), np.array([[self.simulator.setpoint]])])
        
        try:
            solution = np.linalg.solve(M, rhs)
            x_ss = solution[:2]  # state at steady state
            u_ss = solution[2:]  # control input at steady state
        except np.linalg.LinAlgError:
            # fallback if the system is singular
            if hasattr(self, 'prev_x_ss'):  # use cache
                x_ss = self.prev_x_ss
                u_ss = self.prev_u_ss
            else:  # set to zero (assume regulation)
                x_ss = np.zeros((2, 1))
                u_ss = np.zeros((1, 1))
        
        # cache steady state values
        self.prev_x_ss = x_ss
        self.prev_u_ss = u_ss

        # observer dynamics - track error from steady state
        # x_k: estimate of plant state vector x
        dx_k = self.A_cl @ (self.x_k - x_ss) - self.H @ e
        self.x_k += dx_k * self.simulator.solver_dt  # Euler's method (simple)
        
        # control input: u = F @ x (offset by steady-state values)
        u = u_ss + self.F @ (self.x_k - x_ss)
        u = float(u[0][0])  # convert back to scalar

        self.simulator.last_u = u
        return u
"""
