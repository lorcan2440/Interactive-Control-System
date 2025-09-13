import numpy as np

from controllers import ControllerType


class PlantModel:
    def __init__(self, simulator):

        self.simulator = simulator

        self.k12 = 10
        self.k21 = 20
        self.d = 1
    
    def system_dynamics(self, t: float, x: np.ndarray):
        '''
        Defines the system's differential equations. 
        This function is called every `solver_dt` (multiple times per the animation step `dt`).

        Variables:
        - x_1: drug concentration in compartment 1 (state variable 1)
        - x_2: drug concentration in compartment 2 (state variable 2)
        - u: drug injection (control input)
        - w_1: model disturbance in compartment 2
        - w_2: measurement noise
        - y: measurement of compartment 2
        - t: time

        Constants:
        - k_12, k_21: flow rates between compartments
        - d: drug degradation rate

        State space model:
        - x_1' = -(k_12 + d) * x_1 + k_21 * x_2 + u
        - x_2' = k_12 * x_1 - (k_21 + d) * x_2 + w_1
        - y = x_2 + w_2
        '''
        # generate process noise sample
        w1 = np.random.normal(0, self.simulator.w1_stddev)

        match self.simulator.controller_type:
            case ControllerType.MANUAL:
                u = self.simulator.manual_controller.calc_u(self.simulator.y_measured)
            case ControllerType.PID:
                u = self.simulator.pid_controller.calc_u(self.simulator.y_measured)
            case ControllerType.H2:
                u = self.simulator.h2_controller.calc_u(self.simulator.y_measured)
            case _:
                u = 0  # uncontrolled if no valid controller selected
    
        # rates of change with control input and process noise
        dx1 = -(self.k12 + self.d) * x[0] + self.k21 * x[1] + u
        dx2 = self.k12 * x[0] - (self.k21 + self.d) * x[1] + w1
        
        return [dx1, dx2]
    
    def measurement(self, x: np.ndarray):
        '''
        Generate a noisy measurement of the system state.
        '''

        # generate measurement noise sample
        w2 = np.random.normal(0, self.simulator.w2_stddev)  # measurement noise
        y_measured = x[1] + w2
        return y_measured