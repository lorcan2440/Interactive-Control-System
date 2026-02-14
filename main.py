"""Program entry point: create and run the GUI application.
"""

# built-ins
import sys

# external imports
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QTimer

# local imports
from plant import Plant
from controllers import ControllerType, ManualController, OpenLoopController, BangBangController, PIDController
from gui import GUI
from utils import get_logger, TIME_STEPS, PLANT_DEFAULT_PARAMS, GUI_SLIDER_CONFIG, CONTROLLER_PARAMS_LIST


class Simulation(QWidget):

    def __init__(self):

        '''
        Create a simulation of a closed-loop feedback control system. Definitions:

        - The **plant** is the system we wish to control. We can apply inputs to the plant, and measure its output.
        - The **set point** (aka 'reference') is the value that we want our plant's output to go towards.
        - The **error** is the difference between the plant's measured output and the set point.
        - The **controller** receives the error and computes a suitable input to apply to the plant.

        This forms a continuous cycle which iterates through time. A well-designed controller should reduce the 
        error to zero, guiding the plant being controlled towards its set point, without causing the plant to
        'blow up' (lose stability), despite the presence of random disturbances (noise).

        The plant's behaviour is mathematically modelled using a **state**, which in general can be a 
        vector of variables which evolve in time. Disturbances are included in this simulation as 
        additive white noise (random numbers drawn from a Normal distribution with zero mean are added 
        to the plant model).

        Plant variables:
        - x_1: drug concentration in compartment 1 (state variable 1)
        - x_2: drug concentration in compartment 2 (state variable 2)
        - u: drug injection (control input)
        - w_1: model disturbance in compartment 2
        - w_2: measurement noise
        - y: measurement of compartment 2
        - t: time

        Transfer functions (TFs) are the relationships between the inputs and outputs of any system 
        (e.g. the controller, the plant, the whole system), expressed as functions of s in the Laplace transform 
        domain. For s = jω, the TF G(s) is G(jω), which is the frequency response of the system at input frequency ω.
        '''

        super().__init__()  # init base class constructor

        # init logging
        self.logger = get_logger()

        # set time step sizes
        self.dt_int = TIME_STEPS['DT_INT']  # integration time step for solving dynamics
        self.dt_anim = TIME_STEPS['DT_ANIM']  # animation (frame) and control loop time step
        self.dt_window = TIME_STEPS['DT_SLIDING_WINDOW']  # time window size for sliding window plot

        # simulation time
        self.t = 0.0
        # init plant
        self.plant = Plant(dims=2, **PLANT_DEFAULT_PARAMS)

        # init all controllers
        self.manual_controller = ManualController(sim=self, plant=self.plant)
        self.openloop_controller = OpenLoopController(sim=self, plant=self.plant)
        self.bangbang_controller = BangBangController(sim=self, plant=self.plant)
        self.pid_controller = PIDController(sim=self, plant=self.plant)

        # set controller type
        self.controller_type = ControllerType.MANUAL

        # set initial values
        for param in CONTROLLER_PARAMS_LIST + ['y_sp']:
            setattr(self, param, GUI_SLIDER_CONFIG[param]['init'])

        # initialise GUI window
        self.gui = GUI(self)  # the gui and sim can both access each other's attributes and methods
        self.gui.init_gui()

        # start simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # initially stopped
        self.running = False

    def update_frame(self):
        '''
        Update the simulation for one frame of the interactive control loop. 
        This is called by a timer every `self.dt_anim` seconds.
        '''

        # get current time
        t_start = self.t

        # solve dynamics for this frame
        t_span, x_span, y_meas = self.solve_one_frame(t_start)

        # update GUI with new data
        self.gui.update_plots(t_span, x_span, y_meas)

    def solve_one_frame(self, t_start: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Solves the system dynamics for one frame of the interactive control loop, from `t_start`.
        The stop time is `t_start + self.dt_anim`.

        The control input `u` is calculated once at `t_start` using the controller's `calc_u` method,
        based on the value of `y_sp` and `y` at `t_start`, and this value of u is used throughout the frame.

        ### Arguments
        - `t_start`: start time of the frame. Shape: scalar

        ### Returns
        - `tuple[np.ndarray, np.ndarray, np.ndarray]`: arrays of time points, state trajectory, and measurement over this frame (including both endpoints).
        '''

        # if this is the first frame, y has not been set yet, so compute it now
        if not hasattr(self.plant, 'y'):
            self.plant.sample_measurement()
            self.y_0 = self.plant.y

        # get error at start of this frame
        e = self.y_sp - self.plant.y  # shape (1, 1)

        # select current controller and compute control input based on error
        match self.controller_type:
            case ControllerType.MANUAL:
                u = self.manual_controller.calc_u()
            case ControllerType.OPENLOOP:
                u = self.openloop_controller.calc_u()
            case ControllerType.BANGBANG:
                u = self.bangbang_controller.calc_u(e)
            case ControllerType.PID:
                u = self.pid_controller.calc_u(e)
            case _:
                raise ValueError(f'Invalid controller type: {self.controller_type}')

        # update plant control input
        self.plant.u = u

        # solve dynamics
        t_stop = t_start + self.dt_anim
        t_span, x_span = self.plant.integrate_dynamics(t_start=t_start, t_stop=t_stop, dt=self.dt_int)

        # get measurement noise and measurement at t_stop
        self.plant.sample_measurement()

        # advance simulation time
        self.t = t_stop

        return t_span, x_span, self.plant.y


def main():
    app = QApplication(sys.argv)
    sim = Simulation()
    sim.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()