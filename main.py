# built-ins
import sys

# externals
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PyQt6.QtWidgets import QApplication, QWidget, QSlider
from PyQt6.QtCore import Qt, QTimer

# locals
from controllers import (ControllerType, ManualController, OpenLoopController, BangBangController, 
                         PIDController, H2Controller)
from plant import PlantModel
from gui import GUI
from secondary_plots import PlotType


try:
    plt.style.use(r'C:\LibsAndApps\Python config files\proplot_style.mplstyle')
except OSError:
    plt.style.use('ggplot')


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
        
        # call the base class constructor
        super().__init__()

        # time steps
        self.dt = 0.1               # GUI and measurement updates every dt time units
        self.solver_dt = 0.005      # dynamics update every solver_dt time units
        self.graph_window = 5.0     # time units shown on graph

        # graph range
        self.y_lim_minus = -2.0
        self.y_lim_plus = 2.0
        self.u_lim_minus = -5.0
        self.u_lim_plus = 5.0
        self.freq_min = 1e-3
        self.freq_max = 1e4
        self.freq_range = np.logspace(np.log10(self.freq_min), np.log10(self.freq_max), 100)

        # default settings (initial values)
        self.w1_stddev = 0.0                            # process noise standard deviation
        self.w2_stddev = 0.0                            # measurement noise standard deviation
        self.state_x = np.array([0.0, 0.0])             # initial states: x1, x2
        self.y_measured = self.state_x[1]               # initial measured output
        self.controller_type = ControllerType.MANUAL    # default controller type
        
        self.manual_u = 0.0
        self.manual_enabled = True if self.controller_type is ControllerType.MANUAL else False
        self.last_u = 0.0

        self.U_plus = 4.0
        self.U_minus = -4.0

        self.setpoint = 1.0
        self.Kp = 5.0
        self.Ki = 0.0
        self.Kd = 0.0

        self.C1_1 = 0.0
        self.C1_2 = 1.0

        # sliders: set min, max, step size and initial value for each.
        self.slider_configs = {
            'manual_u':  {'min': -5.0,  'max': 5.0,     'step': 0.1,    'init': self.manual_u},
            'setpoint':  {'min': -1.0,  'max': 1.0,     'step': 0.01,   'init': self.setpoint},
            'w1_stddev': {'min': 0.0,   'max': 2.0,     'step': 0.01,   'init': self.w1_stddev},
            'w2_stddev': {'min': 0.0,   'max': 0.5,     'step': 0.01,   'init': self.w2_stddev},
            'U_plus':    {'min': 0.0,   'max': 5.0,     'step': 0.1,    'init': self.U_plus},
            'U_minus':   {'min': -5.0,  'max': 0.0,     'step': 0.1,    'init': self.U_minus},
            'Kp':        {'min': 0.0,   'max': 20.0,    'step': 0.05,   'init': self.Kp},
            'Ki':        {'min': 0.0,   'max': 20.0,    'step': 0.05,   'init': self.Ki},
            'Kd':        {'min': 0.0,   'max': 50.0,    'step': 0.05,   'init': self.Kd},
            'C1_1':      {'min': -3.0,  'max': 3.0,     'step': 0.1,    'init': self.C1_1},
            'C1_2':      {'min': -3.0,  'max': 3.0,     'step': 0.1,    'init': self.C1_2},
        }
        
        # attach plant dynamic model that is being controlled
        self.plant = PlantModel(self)

        # attach controller definitions
        self.manual_controller = ManualController(self, self.plant)
        self.openloop_controller = OpenLoopController(self, self.plant)
        self.bangbang_controller = BangBangController(self, self.plant)
        self.pid_controller = PIDController(self, self.plant)
        self.h2_controller = H2Controller(self, self.plant)
        
        # initialise GUI window
        self.gui = GUI(self)
        self.gui.init_two_plot_layout()
        self.gui.init_settings()

        # start simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(int(self.dt * 1000))  # convert dt to milliseconds

    def update_simulation(self):
        '''
        Simulate the system trajectory between now and the next animation frame.
        '''
        t_span = (0, self.dt)
        t_eval = np.arange(0, self.dt, self.solver_dt)

        # sample process noise
        w1 = np.random.normal(loc=0, scale=self.w1_stddev)

        # Simulate plant dynamics over this time step
        # The control input is computed as part of the dynamics, and may vary throughout the step.
        # The controller only sees the value of y_measured, which is constant throughout the step.
        sol = solve_ivp(self.plant.system_dynamics, t_span, self.state_x, args=(w1,),
                        t_eval=t_eval, method='RK45', max_step=self.solver_dt)

        self.state_x = sol.y[:, -1]  # state vector at the end of the interval

        # sample measurement noise and compute the measured output
        w2 = np.random.normal(loc=0, scale=self.w2_stddev)
        self.y_measured = self.plant.measurement(self.state_x, w2)

        self.update_time_domain_plots()

    def update_time_domain_plots(self):
        '''
        Graph the next animation frame in the time domain. For better frame rate, only the endpoints 
        of each time step are plotted, not the intermediate points (even though they are calculated 
        in the integration step).
        '''

        # update plot data
        self.t_data.append(self.t_data[-1] + self.dt)
        self.y_true_data.append(self.state_x[1])        # true output y = x2
        self.y_measured_data.append(self.y_measured)    # noisy measurement
        self.u_data.append(self.last_u)

        # only retain the most recent data points visible on the graph
        if len(self.t_data) > 1 + self.graph_window / self.dt:
            self.t_data.pop(0)
            self.y_true_data.pop(0)
            self.y_measured_data.pop(0)
            self.u_data.pop(0)

        # use a sliding window of self.graph_window time units
        if self.t_data[-1] > self.graph_window:
            self.ax1.set_xlim(self.t_data[-1] - self.graph_window, self.t_data[-1])
            self.ax2.set_xlim(self.t_data[-1] - self.graph_window, self.t_data[-1])

        # update time domain plots with truncated data
        self.line_y_true.set_xdata(self.t_data)
        self.line_y_true.set_ydata(self.y_true_data)
        self.line_y_true.set_label('$ y_{true} = $' + f' {self.y_true_data[-1]:.3f}')
        self.line_y_measured.set_xdata(self.t_data)
        self.line_y_measured.set_ydata(self.y_measured_data)
        self.ref_line.set_xdata(self.t_data)
        self.ref_line.set_ydata([self.setpoint] * len(self.t_data))

        self.line_u.set_xdata(self.t_data)
        self.line_u.set_ydata(self.u_data)
        self.line_u.set_label('$ u = $' + f' {self.last_u:.3f}')

        # refresh legend and y-axis scale
        self.ax1.legend(loc='lower left')
        self.ax2.legend(loc='lower left')

        self.ax1.set_ylim(min(self.y_lim_minus, min(self.y_measured_data)), 
                          max(self.y_lim_plus, max(self.y_measured_data)))
        self.ax2.set_ylim(min(self.u_lim_minus, min(self.u_data)), 
                          max(self.u_lim_plus, max(self.u_data)))

        self.canvas.draw()

    def update_bode_plots(self):

        oltf_jw = [self.pid_controller.open_loop_tf(complex(0, 1) * w) for w in self.freq_range]
        bode_gains = 20 * np.log10(np.abs(oltf_jw))
        bode_phases = np.angle(oltf_jw, deg=True)
        self.line_bode_gain.set_ydata(bode_gains)
        self.line_bode_phase.set_ydata(bode_phases)

        self.ax3.set_ylim(min(min(bode_gains), -60) - 5, max(max(bode_gains), 60) + 5)
        self.ax4.set_ylim(min(min(bode_phases), -180) - 5, max(max(bode_phases), 0) + 5)

        self.ax3.legend(loc='upper right')
        self.ax4.legend(loc='upper right')

    def update_nyquist_plot(self):

        oltf_jw_plus = [self.pid_controller.open_loop_tf(complex(0, 1) * w) for w in self.freq_range]
        oltf_jw_minus = [self.pid_controller.open_loop_tf(complex(0, -1) * w) for w in self.freq_range]
        nyquist_re_plus = np.real(oltf_jw_plus)
        nyquist_im_plus = np.imag(oltf_jw_plus)
        nyquist_re_minus = np.real(oltf_jw_minus)
        nyquist_im_minus = np.imag(oltf_jw_minus)

        self.line_nyquist_plus.set_xdata(nyquist_re_plus)
        self.line_nyquist_plus.set_ydata(nyquist_im_plus)
        self.line_nyquist_minus.set_xdata(nyquist_re_minus)
        self.line_nyquist_minus.set_ydata(nyquist_im_minus)

        L_plus_1 = self.pid_controller.open_loop_tf(complex(0, 1))
        Lprime_plus_1 = self.pid_controller.open_loop_tf(complex(0, 1.1)) - self.pid_controller.open_loop_tf(complex(0, 0.9))
        Lprime_plus_1 /= abs(Lprime_plus_1)

        L_minus_1 = self.pid_controller.open_loop_tf(complex(0, -1))
        Lprime_minus_1 = self.pid_controller.open_loop_tf(complex(0, -1.1)) - self.pid_controller.open_loop_tf(complex(0, -0.9))
        Lprime_minus_1 /= abs(Lprime_minus_1)

        self.circle_nyquist_plus.set_center((np.real(L_plus_1), np.imag(L_plus_1)))
        self.circle_nyquist_minus.set_center((np.real(L_minus_1), np.imag(L_minus_1)))

        self.ax3.set_xlim(-2.5, 2.5)
        self.ax3.set_ylim(-2.5, 2.5)

        self.ax3.legend(loc='upper right')

    def on_controller_changed(self):
        '''
        Handle controller type change via radio buttons.
        '''
        if self.manual_radio.isChecked():
            self.controller_type = ControllerType.MANUAL
            self.secondary_plot_settings_box.setVisible(False)

        elif self.openloop_radio.isChecked():
            self.controller_type = ControllerType.OPENLOOP
            self.secondary_plot_settings_box.setVisible(False)

        elif self.bangbang_radio.isChecked():
            self.controller_type = ControllerType.BANGBANG
            self.secondary_plot_settings_box.setVisible(False)

        elif self.pid_radio.isChecked():
            self.controller_type = ControllerType.PID
            self.secondary_plot_settings_box.setVisible(True)
            self.pid_controller.reset_memory()  # reset PID integral/last error when changing to PID

        elif self.h2_radio.isChecked():
            self.controller_type = ControllerType.H2
            self.secondary_plot_settings_box.setVisible(False)
            self.h2_controller.reset_memory()

        self.gui.show_controller_settings_box(self.controller_type)

    def on_secondary_plot_changed(self):

        if self.secondary_off_radio.isChecked():
            self.plot_type = PlotType.HIDE
            self.gui.del_plots(keep_time_domain_only=True)
        elif self.bode_radio.isChecked():
            self.plot_type = PlotType.BODE
            self.gui.init_bode_plot()
            self.update_bode_plots()
        elif self.nyquist_radio.isChecked():
            self.plot_type = PlotType.NYQUIST
            self.gui.init_nyquist_plot()
            self.update_nyquist_plot()
        elif self.nichols_radio.isChecked():
            self.plot_type = PlotType.NICHOLS
            self.gui.init_nichols_plot()
            self.update_nichols_plot()
        elif self.root_locus_radio.isChecked():
            self.plot_type = PlotType.ROOTLOCUS
            self.gui.init_root_locus_plot()
    
    def update_setpoint(self, value):
        cfg = self.slider_configs['setpoint']
        self.setpoint = cfg['min'] + value * cfg['step']
        self.setpoint_label.setText(f"Setpoint: {self.setpoint:.3f}")
        # if plot exists, update setpoint immediately
        if hasattr(self, 'ref_line') and len(self.t_data) > 0:
            self.ref_line.set_ydata([self.setpoint] * len(self.t_data))

    def update_w1_stddev(self, value):
        cfg = self.slider_configs['w1_stddev']
        self.w1_stddev = cfg['min'] + value * cfg['step']
        self.w1_stddev_label.setText(f"Process Noise (w1): {self.w1_stddev:.3f}")

    def update_w2_stddev(self, value):
        cfg = self.slider_configs['w2_stddev']
        self.w2_stddev = cfg['min'] + value * cfg['step']
        self.w2_stddev_label.setText(f"Measurement Noise (w2): {self.w2_stddev:.3f}")

    def update_manual_u(self, value):
        cfg = self.slider_configs['manual_u']
        self.manual_u = cfg['min'] + value * cfg['step']
        self.manual_label.setText(f"Manual Control u: {self.manual_u:.3f}")

    def update_manual_enabled(self, state):
        self.manual_enabled = (state == Qt.CheckState.Checked.value)
        self.update_manual_slider_state()

    def update_manual_slider_state(self):
        # Enable/disable slider based on checkbox state
        self.manual_slider.setEnabled(self.manual_enabled)
        self.manual_label.setEnabled(self.manual_enabled)
        
        # Update label to show disabled state
        if self.manual_enabled:
            self.manual_label.setText(f"Manual Control u: {self.manual_u:.3f}")
        else:
            self.manual_label.setText(f"Manual Control u: {self.manual_u:.3f} (Disabled)")

    def update_U_minus(self, value):
        cfg = self.slider_configs['U_minus']
        self.U_minus = cfg['min'] + value * cfg['step']
        self.U_minus_label.setText(f"U_minus: {self.U_minus:.2f}")

    def update_U_plus(self, value):
        cfg = self.slider_configs['U_plus']
        self.U_plus = cfg['min'] + value * cfg['step']
        self.U_plus_label.setText(f"U_plus: {self.U_plus:.2f}")

    def update_Kp(self, value):
        cfg = self.slider_configs['Kp']
        self.Kp = cfg['min'] + value * cfg['step']
        self.Kp_label.setText(f"Kp: {self.Kp:.4f}")
        if self.plot_type is PlotType.BODE:
            self.update_bode_plots()
        elif self.plot_type is PlotType.NYQUIST:
            self.update_nyquist_plot()
        elif self.plot_type is PlotType.NICHOLS:
            self.update_nichols_plot()

    def update_Ki(self, value):
        cfg = self.slider_configs['Ki']
        self.Ki = cfg['min'] + value * cfg['step']
        self.Ki_label.setText(f"Ki: {self.Ki:.4f}")
        if self.plot_type is PlotType.BODE:
            self.update_bode_plots()
        elif self.plot_type is PlotType.NYQUIST:
            self.update_nyquist_plot()
        elif self.plot_type is PlotType.NICHOLS:
            self.update_nichols_plot()

    def update_Kd(self, value):
        cfg = self.slider_configs['Kd']
        self.Kd = cfg['min'] + value * cfg['step']
        self.Kd_label.setText(f"Kd: {self.Kd:.4f}")
        if self.plot_type is PlotType.BODE:
            self.update_bode_plots()
        elif self.plot_type is PlotType.NYQUIST:
            self.update_nyquist_plot()
        elif self.plot_type is PlotType.NICHOLS:
            self.update_nichols_plot()

    def update_C1_1(self, value):
        cfg = self.slider_configs['C1_1']
        self.C1_1 = cfg['min'] + value * cfg['step']
        self.C1_1_label.setText(f"C1_1: {self.C1_1:.2f}")

    def update_C1_2(self, value):
        cfg = self.slider_configs['C1_2']
        self.C1_2 = cfg['min'] + value * cfg['step']
        self.C1_2_label.setText(f"C1_2: {self.C1_2:.2f}")

    def make_slider_from_cfg(self, key: dict[str, float],
                             orientation: Qt.Orientation = Qt.Orientation.Horizontal) -> QSlider:
        '''
        Helper function to make a slider in the GUI with given min, max, step size and initial value.
        
        ### Arguments
        #### Required
        - `key` (dict[str, float]): a dict of {'min': ..., 'max': ..., 'step': ..., 'init': ...}
        #### Optional
        - `orientation` (Qt.Orientation) (default = Qt.Orientation.Horizontal): 
        whether to have the slider vertical or horizontal.
        
        ### Returns
        - `QSlider`: the Qt slider object
        
        ### Raises
        - `ValueError`: if the step size is not a positive value.
        '''        

        # Helper function to make slider from {min, max, step, init}
        cfg = self.slider_configs[key]
        if cfg['step'] <= 0:
            raise ValueError(f"slider step must be > 0 for '{key}'")
        
        # number of steps (integer)
        n_steps = max(1, int(round((cfg['max'] - cfg['min']) / cfg['step'])))
        s = QSlider(orientation)
        s.setMinimum(0)
        s.setMaximum(n_steps)

        # initial integer position
        init_int = int(round((cfg.get('init', cfg['min']) - cfg['min']) / cfg['step']))
        init_int = min(max(init_int, 0), n_steps)
        s.setValue(init_int)
        s.setSingleStep(1)
        s.setPageStep(max(1, n_steps // 10))

        return s


def main():
    app = QApplication(sys.argv)
    sim = Simulation()
    sim.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()