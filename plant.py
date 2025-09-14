import numpy as np

from controllers import ControllerType


class PlantModel:
    def __init__(self, simulator):

        self.simulator = simulator

        self.k12 = 10
        self.k21 = 20
        self.d = 1
    
    def system_dynamics(self, t: float, x: np.ndarray, w1: float = 0) -> tuple[float]:
        '''
        Defines the system's differential equations.
        This function is called every `solver_dt` (multiple times per the animation step `dt`).
        The same disturbance input `w1` is used throughout the animation step.
        The control input `u` is calculated within this function each time it is called.
        '''

        # get error signal - 180 degree phase shift from y_measured due to sign convention
        e = self.simulator.setpoint - self.simulator.y_measured

        # compute control input using the selected controller type
        match self.simulator.controller_type:
            case ControllerType.MANUAL:
                u = self.simulator.manual_controller.calc_u()
            case ControllerType.OPENLOOP:
                u = self.simulator.openloop_controller.calc_u()
            case ControllerType.BANGBANG:
                u = self.simulator.bangbang_controller.calc_u(e)
            case ControllerType.PID:
                u = self.simulator.pid_controller.calc_u(e)
            case ControllerType.H2:
                u = self.simulator.h2_controller.calc_u(e)
            case _:
                u = 0  # uncontrolled if no valid controller selected
    
        # rates of change with control input and process noise
        dx1 = -(self.k12 + self.d) * x[0] + self.k21 * x[1] + u  # d(x_1)/dt
        dx2 = self.k12 * x[0] - (self.k21 + self.d) * x[1] + w1  # d(x_2)/dt
        
        return [dx1, dx2]
    
    def measurement(self, x: np.ndarray, w2: float = 0) -> float:
        '''
        Generate a noisy measurement of the system state `x` using the disturbance `w2`.
        '''

        y_measured = x[1] + w2
        return y_measured