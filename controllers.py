from enum import Enum, auto

import numpy as np
from scipy.linalg import solve_continuous_are


class ControllerType(Enum):
    MANUAL = auto()
    OPENLOOP = auto()
    BANGBANG = auto()
    PID = auto()
    H2 = auto()
    
    def __str__(self):
        """Return the string representation for display purposes"""
        return self.name


class ManualController:
    def __init__(self, simulator, plant):
        '''
        The manual controller allows the user to directly specify the control input.
        No computations are performed; only the slider value is read from the GUI and applied to the plant.
        '''
        self.simulator = simulator
        self.plant = plant
    
    def calc_u(self, _y: float):
        '''
        Manual controller that directly sets the control input.
        '''
        if self.simulator.manual_enabled:
            u = self.simulator.manual_u
        else:
            u = 0.0
            
        self.simulator.last_u = u
        return u


class OpenLoopController:
    def __init__(self, simulator, plant):
        '''
        An open-loop controller (aka feedforward controller) is a simple control law whose control input
        is proportional to the reciprocal of the steady-state gain of the plant.
        
        There is no measuring of the output i.e. no feedback, the block diagram is 'open': 
        the control input depends only on the setpoint.
        The controller response time is limited by the dynamics of the plant.
        In the complete absence of disturbances, the steady-state error will be zero.
        '''
        self.simulator = simulator
        self.plant = plant

    def calc_u(self, _y: float):
        y_sp = self.simulator.setpoint

        # steady state gain of plant in open-loop conditions = G(0)
        plant_ss_gain = self.plant.k12 / (self.plant.d * (self.plant.d + self.plant.k12 + self.plant.k21))

        u = y_sp / plant_ss_gain
        self.simulator.last_u = u
        return u


class BangBangController:
    def __init__(self, simulator, plant):
        '''
        A bang-bang controller (aka 'on-off controller') only has two possible control inputs:

        - u = U_plus,  if y_measured < y_setpoint
        - u = U_minus, if y_measured > y_setpoint

        where U_plus > 0 and U_minus < 0 are constants and are the parameters of the controller.

        This type of controller resembles how a thermostat works. It can lead to chattering (rapid changes of u) 
        near the setpoint if there is measurement noise and/or if the plant process dynamics are fast.
        '''
        self.simulator = simulator
        self.plant = plant

    def calc_u(self, y: float):
        if y < self.simulator.setpoint:
            u = self.simulator.U_plus
        elif y > self.simulator.setpoint:
            u = self.simulator.U_minus
        else:
            u = 0

        self.simulator.last_u = u
        return u


class PIDController:
    def __init__(self, simulator, plant):
        '''
        A PID controller performs three separate operations on the error signal:

        1) P (proportional): multiplies the error by the parameter Kp
        2) I (integral): integrates the error over all previous values and multiplies by the parameter Ki
        3) D (derivative): computes the rate of change of the error and multiplies by the parameter Kd

        The controller gain parameters Kp, Ki and Kd determine the relative weighting of each contribution.
        These are set by sliders in the GUI.

        - A larger Kp tends to give faster responses and a smaller steady state error
        (though Kp alone can typically not eliminate a steady state error)

        - A larger Ki tends to eliminate the steady state error faster
        (though it can lead to oscillations in the step response or even instability if too large)

        - A larger Kd tends to smooth out the oscillations caused by Ki
        (though it can cause noise in the input).
        '''

        self.simulator = simulator
        self.plant = plant

        self.reset_memory()

    def reset_memory(self):
        self.integral = 0.0
        self.prev_error = 0.0
    
    def calc_u(self, y: float):
        '''
        Compute the PID control input based on the measured output y.
        '''
        # error signal e
        error = self.simulator.setpoint - y

        # proportional signal
        u_p = self.simulator.Kp * error
        # integral signal (trapezium rule)
        self.integral += 1/2 * (error + self.prev_error) * self.simulator.solver_dt
        u_i = self.simulator.Ki * self.integral
        # derivative signal
        derivative = (error - self.prev_error) / self.simulator.solver_dt
        u_d = self.simulator.Kd * derivative

        # control input
        u = u_p + u_i + u_d

        self.prev_error = error
        self.simulator.last_u = u
        return u


class H2Controller:
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
    
    def calc_u(self, y: float):
        '''
        Compute the H2 optimal control input based on the measured output y.
        '''

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
            self.F = -B2.T @ X   # state feedback gain
            self.H = Y @ C2.T    # Kalman gain
            self.A_cl = A + B2 @ self.F - self.H @ C2  # closed-loop observer matrix
            
            # cache system matrices for steady-state calculation
            self.A_matrix = A
            self.B2_matrix = B2
            self.C2_matrix = C2
            
            self.h2_gains_computed = True
            self.last_C1 = current_C1  # cache performance vector

            # optimal H2 norm - note that MATLAB omits the sqrt(2 * pi) factor
            self.h2_norm = np.sqrt(2 * np.pi * (np.trace(B1.T @ X @ B1) + np.trace(self.F @ Y @ self.F.T)))

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
        y_measured = np.array([[y]])
        output_error = y_measured - self.C2_matrix @ x_ss
        dx_k = self.A_cl @ (self.x_k - x_ss) - self.H @ output_error
        self.x_k += dx_k * self.simulator.solver_dt  # Euler's method (simple)
        
        # control input: u = F @ x (offset by steady-state values)
        u = u_ss + self.F @ (self.x_k - x_ss)
        u = float(u[0][0])  # convert back to scalar

        self.simulator.last_u = u
        return u