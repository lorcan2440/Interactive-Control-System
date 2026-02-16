# built-ins
from enum import Enum, auto

# external imports
import numpy as np

# local imports
from utils import get_logger


class ControllerType(Enum):
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
        self.calc_ss_gain()

    def calc_ss_gain(self):
        '''
        Calculate the steady-state gain of the plant. G(s) is the transfer function from u to y,
        and the steady-state gain to a step change in u is G(0).
        '''
        A, B, C, D = self.plant.A, self.plant.B, self.plant.C, self.plant.D
        try:
            # use the formula ss_gain = G(0), where G(s) is the transfer function from u to y
            # since G(s) = C @ (sI - A)^(-1) @ B + D, we have G(0) = D - C @ A^(-1) @ B
            # HACK: np.linalg.solve returns A^(-1) @ B with better numerical stability
            self.ss_gain = D - C @ np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.logger.warning('''Plant A matrix is singular: step control inputs lead to unbounded outputs. 
                Setting u = 0 for the open-loop controller.''')
            self.ss_gain = np.inf

    def calc_u(self) -> np.ndarray:
        '''
        Calculates the control input for an open-loop (feedforward) controller.
        This depends only on the current setpoint.

        NOTE: if the plant's A matrix is singular (has an eigenvalue of zero), then the control input will 
        always be zero. This is because there is no finite step input that can produce a steady-state value. 
        In principle, we could instead apply an impulse input for one frame, using the formula: 
        `u = e / (self.sim.dt_frame * C @ B)`, but this requires knowing the error `e` (and hence measurement), 
        which is not allowed for an open-loop controller. Therefore, we choose not to implement this case
        and instead take u = 0.

        ### Returns
        - `u`: control input. Shape: (1, 1)
        '''

        # get setpoint
        y_sp = self.sim.y_sp

        # compute control input
        u = y_sp / self.ss_gain
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
    def __init__(self, sim: object = None, plant: object = None):
        self.sim = sim
        self.plant = plant

        self.reset_memory()

    def reset_memory(self):
        self.error_prev = np.array([[0.0]])
        self.error_integrated = np.array([[0.0]])

    def calc_u(self, e: np.ndarray) -> np.ndarray:
        '''
        Calculate the control input `u` based on the error `e`.

        ### Arguments
        - `e`: error (difference in y: setpoint minus measurement). Shape: (1, 1)

        ### Returns
        - `u`: control input. Shape: (1, 1)
        '''
        if not isinstance(e, np.ndarray) or e.shape != (1, 1):
            raise ValueError('Error must be a single-element np.ndarray with shape (1, 1).')
        
        # get proportional component
        u_p = self.sim.K_p * e

        # get integral component
        self.error_integrated += e * self.sim.dt_anim  # right rectangular method
        u_i = self.sim.K_i * self.error_integrated

        # get derivative component
        e_derivative = (e - self.error_prev) / self.sim.dt_anim  # backward difference method
        u_d = self.sim.K_d * e_derivative

        # update previous error
        self.error_prev = e

        # compute control input
        u = u_p + u_i + u_d
        return u

"""
# NOTE: this still uses the old interface - need to update - leave out for now
# TODO: rebuilt from first principles - do not just copy the below, as there are some suspcious results

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