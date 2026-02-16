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
from utils import get_logger, MAX_SIG_FIGS, LOGGING_ON, TIME_STEPS, PLANT_DEFAULT_PARAMS, GUI_SLIDER_CONFIG, CONTROLLER_PARAMS_LIST


class Simulation(QWidget):

    def __init__(self, *args, **kwargs):

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
        - x: state variables (vector)
        - u: control input (scalar)
        - w_proc: process noise (vector)
        - w_meas: measurement noise (scalar)
        - y: output (scalar)
        - y_meas: measured output (scalar)
        - t: time (scalar)

        Transfer functions (TFs) are the relationships between the inputs and outputs of any system 
        (e.g. the controller, the plant, the whole system), expressed as functions of s in the Laplace transform 
        domain. For s = jω, the TF G(s) is G(jω), which is the frequency response of the system at input frequency ω.
        '''

        super().__init__()  # init base class constructor

        # init logging
        self.logger = get_logger()
        if not LOGGING_ON:
            self.logger.disabled = True

        # set time step sizes
        self.dt_int = TIME_STEPS['DT_INT']  # integration time step for solving dynamics
        self.dt_anim = TIME_STEPS['DT_ANIM']  # animation (frame) and control loop time step
        self.dt_window = TIME_STEPS['DT_SLIDING_WINDOW']  # time window size for sliding window plot

        # present time
        self.t = 0.0

        # check time steps are acceptable
        self.EPS = 10 ** (-1 * MAX_SIG_FIGS)  # small number for use with float comparisons
        if not (self.EPS < self.dt_int <= self.dt_anim <= self.dt_window):
            raise ValueError(f'Invalid time step sizes: simulation requires \n'
                f'EPS={self.EPS} < dt_int={self.dt_int} <= dt_anim={self.dt_anim} <= dt_window={self.dt_window}.')
        
        # init plant
        self.plant = Plant(*args, **(PLANT_DEFAULT_PARAMS if not kwargs else kwargs))
        self.plant.EPS = self.EPS  # pass along epsilon

        # init all controllers
        self.manual_controller = ManualController(sim=self, plant=self.plant)
        self.openloop_controller = OpenLoopController(sim=self, plant=self.plant)
        self.bangbang_controller = BangBangController(sim=self, plant=self.plant)
        self.pid_controller = PIDController(sim=self, plant=self.plant)

        # set controller type
        self.controller_type = ControllerType.MANUAL

        # set initial values
        self.y_sp = np.array([[GUI_SLIDER_CONFIG['y_sp']['init']]])  # shape (1, 1)
        for param in CONTROLLER_PARAMS_LIST:
            setattr(self, param, GUI_SLIDER_CONFIG[param]['init'])

        # init GUI window
        self.gui = GUI(self, dump_logs_on_stop=LOGGING_ON)
        self.gui.init_gui()

        # start simulation ticker
        self.ticker = QTimer()
        self.ticker.timeout.connect(self.update_frame)

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
        t_span, x_span, y_span, y_meas = self.solve_one_frame(t_start)

        # update GUI with new data
        self.gui.update_plots(t_span, x_span, y_span, y_meas)

    def solve_one_frame(self, t_start: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Solves the system dynamics for one frame of the interactive control loop, from `t_start`.
        The stop time is `t_start + self.dt_anim`.

        The control input `u` is calculated once at `t_start` using the controller's `calc_u` method,
        based on the value of `y_sp` and `y_meas` at `t_start`, and this value of u is used throughout the frame.

        ### Arguments
        - `t_start`: start time of the frame. Shape: scalar

        ### Returns
        - `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: arrays of time points, state trajectory, output, and
        measured output over this frame (including both endpoints).
        '''

        # if this is the first frame, y_meas has not been set yet, so compute it now
        if not hasattr(self.plant, 'y_meas'):
            self.plant.sample_measurement()
            self.y_meas_0 = self.plant.y_meas

        # get error at start of this frame (based on noisy measurement)
        e = self.y_sp - self.plant.y_meas  # shape (1, 1)

        # select current controller and compute control input based on error
        match self.controller_type:
            case ControllerType.NONE:
                u = np.array([[0.0]])
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

        #if LOGGING_ON:
        #    self.logger.debug(f'For frame starting at t = {t_start:.5f}: \t used u = {u.item():.10f}, \t y_sp = {self.y_sp.item():.5f}, \t y_meas = {self.plant.y_meas.item():.5f}, \t e = {e.item()}.')

        # solve dynamics
        t_stop = t_start + self.dt_anim
        t_span, x_span = self.plant.integrate_dynamics(t_start=t_start, t_stop=t_stop, dt=self.dt_int)

        # calculate true output (without noise)
        y_span = self.plant.calc_y(x_span, u=np.tile(u, (1, x_span.shape[1])))  # shape (1, num_steps)

        # get measurement noise at end of frame
        self.plant.y_meas = y_span[0, -1] + self.plant.sample_measurement_noise(n=1)  # shape (1, 1)

        # advance simulation time
        self.t = t_stop

        return t_span, x_span, y_span, self.plant.y_meas


def main():
    app = QApplication(sys.argv)
    sim = Simulation()
    sim.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()