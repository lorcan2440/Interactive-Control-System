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

    # TODO: BUG: when C = [1, 1], u blows up for large K_p - why?
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
        cl_stable, cl_z_poles = self.is_closed_loop_stable()
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

        # proportional term
        u_p = self.sim.K_p * e

        # integral term
        self.e_integrated += e * dt
        u_i = self.sim.K_i * self.e_integrated

        # filtered derivative on measurement
        if self.sim.K_d == 0:
            u_d = np.array([[0.0]])
        else:
            # low-pass filter time constant: use user-configured `tau` when available,
            # otherwise fall back to 5x the sampling period
            # NOTE: consider setting this to 0.1x the derivative time constant K_p / K_d
            tau = getattr(self.sim, 'tau', max(5.0 * dt, 1e-6))

            # Tustin's method implementation of the first-order low-pass 
            # filter on the derivative term, with input y and output u_d
            alpha = 1.0 - dt / tau
            alpha = max(min(alpha, 1.0), 0.0)

            u_d = alpha * self.u_d_prev - (self.sim.K_d / tau) * (y_meas - self.y_meas_prev)
            self.u_d_prev = u_d

        # update stored noisy measurement
        self.y_meas_prev = y_meas

        # total control input = P + I + D
        u = u_p + u_i + u_d

        return u
    
    def is_closed_loop_stable(self) -> tuple[bool, np.ndarray]:
        """
        Return whether the sampled-data closed loop is asymptotically stable.

        This check uses a linear P-only approximation of the current controller
        (i.e. it ignores `K_i`, `K_d`, and derivative filtering `tau` for now):

            x_dot = A x + B u,      y = C x
            u[k] = K_p (r[k] - y[k]) = K_p (r[k] - C x[k])

        With zero-order hold over one control period `T = dt_anim`, this becomes
        the discrete-time model

            x[k+1] = A_d x[k] + B_d u[k],
            A_d = exp(A T),
            B_d = integral_0^T exp(A t) dt @ B
                = A^{-1}(A_d - I)B   (used below; valid when A is invertible).

        Substituting the P law gives

            x[k+1] = (A_d - B_d K_p C) x[k] + B_d K_p r[k]
                   = A_cl x[k] + ... .

        Stability is determined by the homogeneous part `x[k+1] = A_cl x[k]`:
        all closed-loop poles/eigenvalues `z_i` of `A_cl` must satisfy `|z_i| < 1`.

        For low-order scalar characteristic polynomials, the Jury criterion 
        gives equivalent inequalities. Here we check the equivalent eigenvalue 
        condition directly.

        Continuous-time analogue: poles `s_i` must satisfy `Re(s_i) < 0`.
        Under exact sampling, `z_i = exp(s_i T)`.
        """

        # get plant matrices
        C = self.plant.C  # shape: (1, dims)

        # TODO: can we also calculate this using the continuous-time closed-loop
        # matrix, then use the "are all eigenvalues in the left half plane?" criterion 
        # for stability (Laplace check instead of Z-transform check)?
        # use the relation z_i = exp(s_i T) to convert between the two

        # TODO: generalise to allowing D != 0

        # TODO: generalise to K_i, K_d != 0, including the effect of tau

        # get discrete-time matrices
        A_d, B_d = self.plant.A_d, self.plant.B_d  # shapes: (dims, dims), (dims, 1)

        # when the loop is closed: u_k = K_p * (y_sp - C x_k), so
        # x_{k+1} = A_cl x_k + B_d @ K_p * y_sp, where A_cl = A_d - B_d @ K_p @ C
        A_cl = A_d - B_d @ (self.sim.K_p * C)  # closed-loop state transition matrix

        # Jury stability test: all discrete-time poles z must have |z| < 1
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
