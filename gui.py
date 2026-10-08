# built-ins
import csv
import copy
from datetime import datetime
from pathlib import Path

# external imports
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QRadioButton, QPushButton, \
    QButtonGroup, QDialog, QDialogButtonBox, QMessageBox, QWidget, QGridLayout, QSpinBox, QTableWidget, \
    QTableWidgetItem, QStyledItemDelegate, QLineEdit, QComboBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, mkPen

# local imports
from plant import IntegratorType
from controllers import ControllerType
from utils import make_slider_from_cfg, PLANT_DEFAULT_PARAMS, MAX_SIG_FIGS, ANIM_SPEED_FACTOR, \
    GUI_SLIDER_CONFIG, CONTROLLER_PARAMS_LIST


pg.setConfigOption('background', '#222222')
pg.setConfigOption('foreground', '#DDDDDD')


class GUI:

    # TODO: add the LaTeX equation (provided in the SVG image at media/state_space_model_black.svg)
    # at the top of the change plant model dialog. If possible, detect whether the dialog box
    # is in dark mode or light mode and invert the colors of the SVG dynamically to white for dark mode.
    # This could be done by searching and replacing "stroke="#000000" fill="#000000"" in the SVG string 
    # (only occurs once) with "stroke="#FFFFFF" fill="#FFFFFF" when in dark mode.
    # TODO: add a checkbox in the PID parameters box to enable/disable anti-windup:
    # if checked, show a 'u_sat' slider for the user to set the saturation limit for |u|
    # TODO: add buttons under the PID parameters row to set Kp, Ki, Kd based on IAE, ITAE, 
    # Ziegler-Nichols, Cohen-Coon, and pole placement, using functions implemented in controllers.py PIDController
    # TODO: add a checkbox in the PID parameters box to enable/disable filtering on the derivative:
    # if unchecked, the tau slider should be greyed out
    # need to edit the function in controllers.py to respect this setting
    # TODO: add a Bode plot (shown to the right of the graphs) for the PID controller
    # TODO: add a Nyquist plot (shown to the right of the graphs) for the PID controller
    # TODO: add a Nichols plot (shown to the right of the graphs) for the PID controller
    # TODO: add a root-locus plot (shown to the right of the graphs) for the PID controller

    def __init__(self, sim: object, dump_logs_on_stop: bool = False):

        self.sim = sim
        self.logger = self.sim.logger
        self.dump_logs_on_stop = dump_logs_on_stop

        self.clear_buffers()

        # buffer size constants - rounding accounts for cases when time steps are not multiples of each other
        self.n_int_per_frame = int(np.ceil(round(self.sim.dt_anim / self.sim.dt_int, MAX_SIG_FIGS))) + 1
        self.n_frame_per_window = int(np.ceil(round(self.sim.dt_window / self.sim.dt_anim, MAX_SIG_FIGS))) + 1
        self.n_int_per_window = int(np.ceil(round(self.sim.dt_window / self.sim.dt_int, MAX_SIG_FIGS))) + 1

        # controller UI bookkeeping
        self.sim.controller_type = ControllerType.MANUAL
        self.controller_param_widgets = {}
        self.is_csv_logging = False
        self.csv_log_file = None
        self.csv_writer = None
        self.csv_log_path = None

    def clear_buffers(self):

        # set empty plotting buffers
        self.t_data = np.array([])
        self.x_data = np.array([[] for _ in range(self.sim.plant.dims)])
        self.y_data = np.array([])
        self.y_meas = np.array([])
        self.y_sp_data = np.array([])
        self.u_data = np.array([])

    def clear_graph_traces(self):

        # fully clear graphics layout so old PlotItems/axes are removed
        if hasattr(self, 'win'):
            self.win.clear()

        # states and measurement plot
        self.plot_x = self.win.addPlot(row=0, col=0, title='States (x), Measurement (y_meas) and Setpoint (y_sp)')
        self.plot_x.setAutoVisible(x=False)  # turn off x-axis auto-scaling
        self.plot_x.addLegend()

        # init curves for each state variable
        for i in range(1, self.sim.plant.dims + 1):
            curve_x_i = self.plot_x.plot(pen=pg.intColor(i - 1, hues=self.sim.plant.dims, values=77), name=f'x_{i}')
            curve_x_i.setVisible(False)  # hide all state variables initially
            setattr(self, f'curve_x_{i}', curve_x_i)

        # init curve for measurement
        self.curve_y_meas = self.plot_x.plot(pen=None, symbol='o', 
            symbolPen=mkPen(color='white'), symbolSize=4, symbolBrush=0.2, name='y_meas')
        # init curve for true output y (computed from state-space matrices)
        self.curve_y = self.plot_x.plot(pen=mkPen(color='white'), name='y')
        
        # init curve for setpoint - uses Step Mode (piecewise constant across a frame)
        self.curve_y_sp = self.plot_x.plot(stepMode='left', 
            pen=mkPen(color='#777777', style=Qt.PenStyle.DashLine), name='y_sp')

        # control input plot
        self.plot_u = self.win.addPlot(row=1, col=0, title='Control input (u)')
        self.plot_u.setAutoVisible(x=False)  # turn off x-axis auto-scaling

        # init curve for control input - uses Step Mode (piecewise constant across a frame)
        self.curve_u = self.plot_u.plot(stepMode='left', pen=mkPen(color='green'))

    def init_gui(self):

        # build layout onto the Simulation QWidget
        main_layout = QVBoxLayout()

        ## 1) top area - graphs

        self.win = GraphicsLayoutWidget()  # from PyQtGraph
        self.clear_graph_traces()  # init empty graphs and curves

        main_layout.addWidget(self.win, stretch=1)

        ## 2) first row under graphs - controller selection

        # start/stop button
        first_row_hbox = QHBoxLayout()
        self.start_stop_button = QPushButton('Start')
        self.start_stop_button.clicked.connect(self.toggle_start_stop)
        first_row_hbox.addWidget(self.start_stop_button)

        # reset simulation button
        self.reset_button = QPushButton('Reset')
        self.reset_button.setToolTip('Reset the simulation to its initial conditions')
        self.reset_button.clicked.connect(self.reset_simulation)
        first_row_hbox.addWidget(self.reset_button)

        # start/stop CSV logging button
        self.logging_button = QPushButton('Start Logging')
        self.logging_button.clicked.connect(self.toggle_csv_logging)
        first_row_hbox.addWidget(self.logging_button)

        # change plant model button
        self.change_plant_button = QPushButton('Change plant model')
        self.change_plant_button.clicked.connect(self.open_change_plant_dialog)
        first_row_hbox.addWidget(self.change_plant_button)
        first_row_hbox.addStretch()

        # controller selection box
        controller_buttons_box = QGroupBox('Controller Selection')
        controller_buttons_box_layout = QHBoxLayout()
        self.controller_buttons_group = QButtonGroup()

        # controller selection radio buttons (None first)
        self.radio_none = QRadioButton('None')
        self.radio_manual = QRadioButton('Manual')
        self.radio_openloop = QRadioButton('Open Loop')
        self.radio_bangbang = QRadioButton('Bang-Bang')
        self.radio_pid = QRadioButton('PID')
        self.radio_h2 = QRadioButton('H2')
        self.radio_hinf = QRadioButton('H∞')
        self.radio_mpc = QRadioButton('MPC')

        self.controller_buttons_group.addButton(self.radio_none)
        self.controller_buttons_group.addButton(self.radio_manual)
        self.controller_buttons_group.addButton(self.radio_openloop)
        self.controller_buttons_group.addButton(self.radio_bangbang)
        self.controller_buttons_group.addButton(self.radio_pid)
        self.controller_buttons_group.addButton(self.radio_h2)
        self.controller_buttons_group.addButton(self.radio_hinf)
        self.controller_buttons_group.addButton(self.radio_mpc)

        controller_buttons_box_layout.addWidget(self.radio_none)
        controller_buttons_box_layout.addWidget(self.radio_manual)
        controller_buttons_box_layout.addWidget(self.radio_openloop)
        controller_buttons_box_layout.addWidget(self.radio_bangbang)
        controller_buttons_box_layout.addWidget(self.radio_pid)
        controller_buttons_box_layout.addWidget(self.radio_h2)
        controller_buttons_box_layout.addWidget(self.radio_hinf)
        controller_buttons_box_layout.addWidget(self.radio_mpc)

        controller_buttons_box.setLayout(controller_buttons_box_layout)
        first_row_hbox.addWidget(controller_buttons_box)
        main_layout.addLayout(first_row_hbox)

        ## 3) second row under graphs - setpoint
    
        second_hbox = QHBoxLayout()

        sp_group = QGroupBox('Setpoint (reference)')
        sp_layout = QVBoxLayout()
        sp_cfg = GUI_SLIDER_CONFIG['y_sp']
        container, slider, val_label = make_slider_from_cfg('y_sp', 'Setpoint')
        slider.valueChanged.connect(self.on_setpoint_slider_changed)

        sp = sp_cfg.get('init', sp_cfg['min'])
        val_label.setText(f"Setpoint: {sp:.2f}")
        sp_layout.addWidget(container)
        self.slider = slider
        self.sp_label = val_label
        sp_group.setLayout(sp_layout)
        second_hbox.addWidget(sp_group, stretch=3)

        main_layout.addLayout(second_hbox)

        ## 4) third row under graphs - controller parameters (dynamically generated)
        self.params_box = QGroupBox('Controller Parameters')
        self.params_layout = QHBoxLayout()
        self.params_box.setLayout(self.params_layout)
        main_layout.addWidget(self.params_box)

        self.sim.setLayout(main_layout)

        # connect controller radio buttons
        # HACK: on_controller_selected(...) is only run if button is toggled on
        self.radio_none.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.NONE))
        self.radio_manual.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.MANUAL))
        self.radio_openloop.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.OPENLOOP))
        self.radio_bangbang.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.BANGBANG))
        self.radio_pid.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.PID))
        self.radio_h2.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.H2))
        self.radio_hinf.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.HINF))
        self.radio_mpc.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.MPC))

        # initial controller selection and params
        self.radio_manual.setChecked(True)
        self.sim.manual_u = GUI_SLIDER_CONFIG['manual_u']['init']
        self.set_controller(ControllerType.MANUAL)
        self.build_controller_params(ControllerType.MANUAL)

    def update_plots(self, t_span: np.ndarray, x_span: np.ndarray, y_span: np.ndarray, y_meas: np.ndarray):
        # t_span: array of times for this frame. Shape: (n,)
        # x_span: state trajectory for this frame. Shape: (2, n)
        # y_span: true output trajectory for this frame. Shape: (1, n)
        # y_meas: latest measurement. Shape: (1, 1)

        # compute setpoint numeric value
        sp_cfg = GUI_SLIDER_CONFIG['y_sp']
        y_sp_val = sp_cfg['min'] + int(self.slider.value()) * sp_cfg['step']

        # u used during this frame (constant across this frame)
        u_last = float(self.sim.plant.u.item())

        # current measured output (scalar)
        y_meas_last = float(y_meas.item())

        if self.t_data.size == 0:
            # first frame: take full time span and states
            u_0 = self.sim.plant.u_0.item()
            y_meas_0 = self.sim.y_meas_0.item()
            self.t_data = t_span.copy()  # shape: (n_int_per_frame,)
            self.x_data = x_span.copy()  # shape: (dims, n_int_per_frame)
            self.y_data = y_span.copy()  # shape: (1, n_int_per_frame)
            self.t_meas = np.array([float(t_span[0]), float(t_span[-1])])  # shape: (2,)
            self.y_meas = np.array([y_meas_0, y_meas_last])  # shape: (2,)
            self.u_data = np.array([u_0, u_0])  # shape: (2,)
            self.y_sp_data = np.array([y_sp_val, y_sp_val])  # shape: (2,)
        else:
            # subsequent frames: append times and state trajectory for this frame
            # get indices to retain from previous data (within the window)
            i_data = max(0, self.t_data.size - self.n_int_per_window + self.n_int_per_frame - 1)
            i_meas = max(0, self.t_meas.size - self.n_frame_per_window + 1)
            # append new data (skip first value of most recent frame to avoid duplication)
            self.t_data = np.concatenate((self.t_data[i_data:], t_span[1:]), axis=0)
            self.x_data = np.concatenate((self.x_data[:, i_data:], x_span[:, 1:]), axis=1)
            self.y_data = np.concatenate((self.y_data[:, i_data:], y_span[:, 1:]), axis=1)
            self.t_meas = np.concatenate((self.t_meas[i_meas:], np.array([float(t_span[-1])])), axis=0)
            self.y_meas = np.concatenate((self.y_meas[i_meas:], np.array([y_meas_last])), axis=0)
            self.u_data = np.concatenate((self.u_data[i_meas:], np.array([u_last])), axis=0)
            self.y_sp_data = np.concatenate((self.y_sp_data[i_meas:], np.array([y_sp_val])), axis=0)
        
        # update curves
        # plot state variables
        for i in range(1, self.sim.plant.dims + 1):
            curve_x_i = getattr(self, f'curve_x_{i}')
            curve_x_i.setData(self.t_data, self.x_data[i - 1])
        # plot measurement, setpoint, and control input
        self.curve_y.setData(self.t_data, self.y_data.flatten())
        self.curve_y_meas.setData(self.t_meas, self.y_meas)
        self.curve_y_sp.setData(self.t_meas, self.y_sp_data)
        self.curve_u.setData(self.t_meas, self.u_data)

        # update x-axis ranges to this window
        self.plot_x.setXRange(max(0, self.t_data[-1] - self.sim.dt_window), 
                              max(self.sim.dt_window, self.t_data[-1]))
        self.plot_u.setXRange(max(0, self.t_data[-1] - self.sim.dt_window), 
                              max(self.sim.dt_window, self.t_data[-1]))

        if self.is_csv_logging:
            self.write_to_csv_and_log()

    ## UI callbacks

    def reset_simulation(self):
        self.sim.reset()
        self.start_stop_button.setText('Start')
        self.clear_buffers()
        self.clear_graph_traces()

    def toggle_csv_logging(self):
        if not self.is_csv_logging:
            self.start_csv_logging()
        else:
            self.stop_csv_logging()

    def start_csv_logging(self):
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.csv_log_path = output_dir / f'simulation_log_{timestamp}.csv'

        try:
            self.csv_log_file = open(self.csv_log_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_log_file)
            self.csv_writer.writerow(['t', 'u', 'x', 'y', 'y_sp', 'e'])
            self.csv_log_file.flush()
        except OSError as e:
            self.csv_log_file = None
            self.csv_writer = None
            self.csv_log_path = None
            QMessageBox.warning(self.sim, 'Failed to start logging', str(e))
            return

        self.is_csv_logging = True
        self.logging_button.setText('Stop Logging')

    def stop_csv_logging(self):
        if self.csv_log_file is not None:
            try:
                self.csv_log_file.close()
            except OSError:
                pass
        self.is_csv_logging = False
        self.csv_writer = None
        self.csv_log_file = None
        self.logging_button.setText('Start Logging')

    def write_to_csv_and_log(self):
        if self.csv_writer is None or self.csv_log_file is None:
            return

        t = float(self.t_data[-1])
        u = float(self.u_data[-1])
        x = self.x_data[:, -1].tolist()
        y = float(self.y_data[0, -1])
        y_sp = float(self.y_sp_data[-1])
        e = y_sp - y

        self.csv_writer.writerow([f'{t:.6f}', f'{u:.6f}', f'[{x[0]:.6f}, {x[1]:.6f}]', f'{y:.6f}', f'{y_sp:.6f}', f'{e:.6f}'])
        self.csv_log_file.flush()

    def toggle_start_stop(self):
        # toggle the simulation ticker on and off
        if not self.sim.running:  # start
            # ticker times out (calls the update) every real dt_anim / ANIM_SPEED_FACTOR seconds
            self.sim.wall_time_prev = None
            self.sim.sim_time_remainder = 0.0
            self.sim.ticker.start(int(self.sim.dt_anim * 1000 / ANIM_SPEED_FACTOR))
            self.sim.running = True
            self.start_stop_button.setText('Stop')
        else:  # stop
            if self.dump_logs_on_stop:
                self.logger.info(f'Stopped: \n'
                    f't_data: {self.t_data}\n'
                    f'x_data: {self.x_data}\n'
                    f'y_data: {self.y_data}\n'
                    f't_meas: {self.t_meas}\n'
                    f'y_meas: {self.y_meas}\n'
                    f'u_data: {self.u_data}\n'
                    f'y_sp_data: {self.y_sp_data}\n'
                    f'shapes: {self.t_data.shape, self.x_data.shape, self.y_data.shape, self.t_meas.shape}\n'
                    f'{self.y_meas.shape, self.u_data.shape, self.y_sp_data.shape}\n\n')
            self.sim.ticker.stop()
            self.sim.wall_time_prev = None
            self.sim.sim_time_remainder = 0.0
            self.sim.running = False
            self.start_stop_button.setText('Start')

    def on_setpoint_slider_changed(self, val: int):
        # update setpoint based on slider value
        cfg = GUI_SLIDER_CONFIG['y_sp']
        sp = cfg['min'] + val * cfg['step']
        self.sp_label.setText(f'{sp:.2f}')
        self.sim.y_sp = np.array([[sp]])

    def on_controller_selected(self, controller_type: ControllerType):
        if controller_type is not self.sim.controller_type:
            self.sim.controller_type = controller_type
            self.set_controller(controller_type)
            self.build_controller_params(controller_type)

    def add_param(self, key: str, display_name: str = None, cfg: dict[str, float] = None):
        # helper: create a controller parameter slider row
        container, slider, val_label = make_slider_from_cfg(key, display_name, cfg=cfg)
        slider.valueChanged.connect(lambda v, k=key: self.on_controller_param_changed(k, v))
        self.params_layout.addWidget(container)
        self.controller_param_widgets[key] = (slider, val_label)

    @staticmethod
    def get_controller_param_config(key: str) -> dict[str, float]:
        if key == 'MPC_N':
            return GUI_SLIDER_CONFIG['MPC_N']
        if key.startswith('H2_C1_x'):
            return GUI_SLIDER_CONFIG['H2_C1_x']
        if key == 'H2_C1_u':
            return GUI_SLIDER_CONFIG['H2_C1_u']
        if key.startswith('Hinf_C1_x'):
            return GUI_SLIDER_CONFIG['Hinf_C1_x']
        if key == 'Hinf_C1_u':
            return GUI_SLIDER_CONFIG['Hinf_C1_u']
        return GUI_SLIDER_CONFIG[key]

    def build_controller_params(self, controller_type: ControllerType):
        
        # clear current controller params box
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.controller_param_widgets.clear()

        match controller_type:
            case ControllerType.NONE:
                lbl = QLabel('No controller: u = 0')
                self.params_layout.addWidget(lbl)
                return
            case ControllerType.MANUAL:
                self.add_param('manual_u', 'Manual u')
            case ControllerType.OPENLOOP:
                lbl = QLabel('Open-loop controller: no parameters')
                self.params_layout.addWidget(lbl)
            case ControllerType.BANGBANG:
                self.add_param('U_minus', 'U_minus')
                self.add_param('U_plus', 'U_plus')
            case ControllerType.PID:
                self.add_param('K_p', 'K_p')
                self.add_param('K_i', 'K_i')
                self.add_param('K_d', 'K_d')
                self.add_param('tau', 'tau')

                # reset PID memory button
                reset_btn = QPushButton('Reset memory')
                reset_btn.setToolTip('Reset PID integrator and derivative history')
                reset_btn.clicked.connect(self.sim.pid_controller.reset_memory)
                self.params_layout.addWidget(reset_btn)
            case ControllerType.H2:
                for i in range(self.sim.plant.dims):
                    key = f'H2_C1_x{i + 1}'
                    if not hasattr(self.sim, key):
                        setattr(self.sim, key, GUI_SLIDER_CONFIG['H2_C1_x']['init'])
                    self.add_param(key, key, GUI_SLIDER_CONFIG['H2_C1_x'])
                if not hasattr(self.sim, 'H2_C1_u'):
                    self.sim.H2_C1_u = GUI_SLIDER_CONFIG['H2_C1_u']['init']
                self.add_param('H2_C1_u', 'H2_C1_u', GUI_SLIDER_CONFIG['H2_C1_u'])
            case ControllerType.HINF:
                for i in range(self.sim.plant.dims):
                    key = f'Hinf_C1_x{i + 1}'
                    if not hasattr(self.sim, key):
                        setattr(self.sim, key, GUI_SLIDER_CONFIG['Hinf_C1_x']['init'])
                    self.add_param(key, key, GUI_SLIDER_CONFIG['Hinf_C1_x'])
                if not hasattr(self.sim, 'Hinf_C1_u'):
                    self.sim.Hinf_C1_u = GUI_SLIDER_CONFIG['Hinf_C1_u']['init']
                self.add_param(
                    'Hinf_C1_u', 'Hinf_C1_u', GUI_SLIDER_CONFIG['Hinf_C1_u']
                )
            case ControllerType.MPC:
                self.add_param('MPC_N', 'Horizon N', GUI_SLIDER_CONFIG['MPC_N'])
                model_button = QPushButton('Set internal plant model')
                model_button.clicked.connect(self.open_mpc_model_dialog)
                self.params_layout.addWidget(model_button)
                cost_button = QPushButton('Set optimisation function')
                cost_button.clicked.connect(self.open_mpc_cost_dialog)
                self.params_layout.addWidget(cost_button)
                constraint_button = QPushButton('Set optimisation constraints')
                constraint_button.clicked.connect(self.open_mpc_constraint_dialog)
                self.params_layout.addWidget(constraint_button)

        # set slider positions to current values
        for key, (slider, val_label) in self.controller_param_widgets.items():
            cfg = self.get_controller_param_config(key)
            current_val = float(getattr(self.sim, key, cfg.get('init', cfg['min'])))
            pos = int(round((current_val - cfg['min']) / cfg['step']))
            pos = min(max(pos, 0), int(round((cfg['max'] - cfg['min']) / cfg['step'])))
            slider.setValue(pos)
            val_label.setText(f"{current_val:.2f}")

    def on_controller_param_changed(self, key: str, int_pos: int):
        cfg = self.get_controller_param_config(key)
        val = cfg['min'] + int_pos * cfg['step']
        _, val_label = self.controller_param_widgets.get(key, (None, None))
        if val_label is not None:
            val_label.setText(f"{val:.2f}")

        if (
            key in CONTROLLER_PARAMS_LIST
            or key.startswith('H2_C1_')
            or key.startswith('Hinf_C1_')
            or key == 'MPC_N'
        ):
            setattr(self.sim, key, val)
            if key == 'MPC_N':
                self.sim.mpc_controller.ensure_horizon(int(val))

    def set_controller(self, controller_type: ControllerType):
        # set the simulation controller type and perform any needed setup
        match controller_type:
            case ControllerType.NONE:
                self.sim.controller_type = ControllerType.NONE
            case ControllerType.MANUAL:
                self.sim.controller_type = ControllerType.MANUAL
            case ControllerType.BANGBANG:
                self.sim.controller_type = ControllerType.BANGBANG
            case ControllerType.OPENLOOP:
                self.sim.controller_type = ControllerType.OPENLOOP
            case ControllerType.PID:
                self.sim.controller_type = ControllerType.PID
                self.sim.pid_controller.reset_memory()
            case ControllerType.H2:
                self.sim.controller_type = ControllerType.H2
                self.sim.h2_controller.reset_memory()
            case ControllerType.HINF:
                self.sim.controller_type = ControllerType.HINF
                self.sim.hinf_controller.reset_memory()
            case ControllerType.MPC:
                self.sim.controller_type = ControllerType.MPC
                self.sim.mpc_controller.reset_memory()

    def _show_mpc_dialog(self, dialog):
        """Pause simulation during a modal MPC settings dialog."""
        was_running = getattr(self.sim, 'running', False)
        if was_running:
            self.sim.ticker.stop()
            self.sim.running = False
        try:
            dialog.exec()
        finally:
            if was_running:
                self.sim.wall_time_prev = None
                self.sim.sim_time_remainder = 0.0
                self.sim.ticker.start(
                    int(self.sim.dt_anim * 1000 / ANIM_SPEED_FACTOR)
                )
                self.sim.running = True

    def open_mpc_model_dialog(self):
        """Edit the MPC internal plant model or copy the true plant model."""
        dialog = QDialog(self.sim)
        dialog.setWindowTitle('Set internal MPC plant model')
        layout = QVBoxLayout(dialog)
        widget = StateSpaceMatrixInput(
            parent=dialog,
            initial_dims=self.sim.plant.dims,
            show_noise_matrices=False,
            show_integrator_controls=False,
            allow_dimension_change=False,
        )
        widget.set_table_ABCD(
            self.sim.mpc_controller.A_hat,
            self.sim.mpc_controller.B_hat,
            self.sim.mpc_controller.C_hat,
            self.sim.mpc_controller.D_hat,
        )
        set_true_button = QPushButton('Set to true plant model')
        set_true_button.clicked.connect(lambda: widget.set_table_ABCD(
            self.sim.plant.A, self.sim.plant.B, self.sim.plant.C, self.sim.plant.D
        ))
        layout.addWidget(set_true_button)
        layout.addWidget(widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(lambda: self._accept_mpc_model(widget, dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self._show_mpc_dialog(dialog)

    def _accept_mpc_model(self, widget, dialog):
        try:
            self.sim.mpc_controller.set_model_matrices(*widget.get_table_ABCD())
        except ValueError as error:
            QMessageBox.warning(dialog, 'Invalid internal model', str(error))
            return
        dialog.accept()

    def open_mpc_cost_dialog(self):
        """Edit stage and terminal MPC value-function weights."""
        self.sim.mpc_controller.ensure_horizon(int(self.sim.MPC_N))
        dialog = MPCIndexedSettingsDialog(
            parent=self.sim,
            horizon=int(self.sim.MPC_N),
            state_count=self.sim.plant.dims,
            values=self.sim.mpc_controller.costs,
            settings_type='cost',
        )
        self._show_mpc_dialog(dialog)
        if dialog.result() == QDialog.DialogCode.Accepted:
            self.sim.mpc_controller.costs = dialog.values

    def open_mpc_constraint_dialog(self):
        """Edit stage and terminal MPC inequality constraints."""
        self.sim.mpc_controller.ensure_horizon(int(self.sim.MPC_N))
        dialog = MPCIndexedSettingsDialog(
            parent=self.sim,
            horizon=int(self.sim.MPC_N),
            state_count=self.sim.plant.dims,
            values=self.sim.mpc_controller.constraints,
            settings_type='constraint',
        )
        self._show_mpc_dialog(dialog)
        if dialog.result() == QDialog.DialogCode.Accepted:
            self.sim.mpc_controller.constraints = dialog.values

    def open_change_plant_dialog(self):
        """Show a dialog allowing the user to edit the plant state-space matrices.

        The simulation ticker is paused while the dialog is open and resumed
        if it was running before.
        """
        was_running = getattr(self.sim, 'running', False)
        try:
            if was_running:
                try:
                    self.sim.ticker.stop()
                except Exception:
                    pass
                self.sim.running = False

            dialog = QDialog(self.sim)
            dialog.setWindowTitle('Change plant model')
            dlg_layout = QVBoxLayout()

            widget = StateSpaceMatrixInput(parent=dialog, initial_dims=self.sim.plant.dims)
            widget.set_use_ode_mode(getattr(self.sim, 'use_ode_mode', False))
            widget.set_integrator_method(getattr(self.sim, 'integrator_method', IntegratorType.EULER_MARUYAMA))

            # pre-fill with current plant matrices if available
            A, B, C, D = self.sim.plant.A, self.sim.plant.B, self.sim.plant.C, self.sim.plant.D
            widget.set_table_ABCD(np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D))
            Q, R = self.sim.plant.Q, self.sim.plant.R
            widget.set_table_QR(np.asarray(Q), np.asarray(R))

            dlg_layout.addWidget(widget)

            eig_label = QLabel('Eigenvalues:')
            dlg_layout.addWidget(eig_label)

            def _update_eigs(*_):
                try:
                    A_cur, _, _, _ = widget.get_table_ABCD()
                except Exception:
                    eig_label.setText('Eigenvalues: (invalid)')
                    return
                try:
                    vals = np.linalg.eigvals(A_cur)
                    # format to 5 significant figures
                    vals_str = ', '.join([f'{v:.5g}' for v in vals])
                    eig_label.setText(f'Eigenvalues: {vals_str}')
                except Exception:
                    eig_label.setText('Eigenvalues: (error)')

            # update eigenvalues when A is edited or dims change
            widget.table_A.itemChanged.connect(_update_eigs)
            widget.spin.valueChanged.connect(_update_eigs)

            # initialise eigenvalue display
            _update_eigs()

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            dlg_layout.addWidget(buttons)
            dialog.setLayout(dlg_layout)

            buttons.accepted.connect(lambda: self.on_accept_change_plant(widget, dialog))
            buttons.rejected.connect(dialog.reject)

            dialog.exec()

        finally:
            # resume simulation if it was running before
            if was_running:
                try:
                    self.sim.wall_time_prev = None
                    self.sim.sim_time_remainder = 0.0
                    self.sim.ticker.start(int(self.sim.dt_anim * 1000 / ANIM_SPEED_FACTOR))
                except Exception:
                    pass
                self.sim.running = True

    def on_accept_change_plant(self, widget, dialog):
        # called when "OK" is clicked in the change plant dialog box

        A_new, B_new, C_new, D_new = widget.get_table_ABCD()
        Q_new, R_new = widget.get_table_QR()

        try:
            # apply new matrices to plant
            self.sim.plant.set_all_arrays(A_new, B_new, C_new, D_new, Q_new, R_new)

        except ValueError as e:
            if str(e) == "State-space matrices have incorrect dimensions.":
                # dims was changed: attempt to set the plant to the new size,
                # reset buffers and rebuild plots so the simulation can resume.
                new_dims = A_new.shape[0]
                self.logger.info(f'Plant dimensions changed: {self.sim.plant.dims} -> {new_dims}. Re-initialising plant and clearing buffers.')

                # update plant dimension and initial state shapes
                self.sim.plant.dims = int(new_dims)
                self.sim.plant.x_0 = np.zeros((new_dims, 1))
                self.sim.plant.x = self.sim.plant.x_0.copy()
                self.sim.plant.u_0 = getattr(self.sim.plant, 'u_0', np.array([[0.0]]))
                self.sim.plant.u = self.sim.plant.u_0.copy()

            if str(e) in ("Noise matrices Q and R have incorrect dimensions.", 
                          "Process noise matrix Q must be symmetric.",
                          "Noise matrices Q and R must be positive semidefinite."):
                QMessageBox.warning(dialog, 'Invalid noise matrices', str(e))
                return

            # apply the new state-space matrices (and recalculate cached values in the plant)
            self.sim.plant.set_all_arrays(A_new, B_new, C_new, D_new, Q_new, R_new)

            # clear data buffers and graph areas
            self.clear_buffers()
            self.clear_graph_traces()

        # for controllers that use plant matrices, recalculate any of their needed params
        self.sim.pid_controller.reset_memory()
        self.sim.h2_controller.reset_memory()
        for i in range(self.sim.plant.dims):
            key = f'H2_C1_x{i + 1}'
            if not hasattr(self.sim, key):
                setattr(self.sim, key, GUI_SLIDER_CONFIG['H2_C1_x']['init'])
        if not hasattr(self.sim, 'H2_C1_u'):
            self.sim.H2_C1_u = GUI_SLIDER_CONFIG['H2_C1_u']['init']
        for i in range(self.sim.plant.dims):
            key = f'Hinf_C1_x{i + 1}'
            if not hasattr(self.sim, key):
                setattr(self.sim, key, GUI_SLIDER_CONFIG['Hinf_C1_x']['init'])
        if not hasattr(self.sim, 'Hinf_C1_u'):
            self.sim.Hinf_C1_u = GUI_SLIDER_CONFIG['Hinf_C1_u']['init']
        if self.sim.controller_type is ControllerType.H2:
            self.build_controller_params(ControllerType.H2)
        elif self.sim.controller_type is ControllerType.HINF:
            self.sim.hinf_controller.reset_memory()
            self.build_controller_params(ControllerType.HINF)
        if self.sim.mpc_controller.A_hat.shape != self.sim.plant.A.shape:
            self.sim.mpc_controller.reset_for_state_dimension()
        if self.sim.controller_type is ControllerType.MPC:
            self.sim.mpc_controller.reset_memory()
            self.build_controller_params(ControllerType.MPC)
        self.sim.use_ode_mode = widget.use_ode_mode
        self.sim.integrator_method = widget.get_integrator_method()

        dialog.accept()


class NumericMatrixInput(QWidget):
    """Editable finite-valued matrix table used by the MPC settings dialogs."""

    def __init__(self, title, initial_matrix, allow_row_count=False, parent=None):
        super().__init__(parent)
        self.allow_row_count = allow_row_count
        self.editor_layout = QVBoxLayout(self)
        self.editor_layout.setContentsMargins(0, 0, 0, 0)
        self.editor_layout.addWidget(QLabel(title))

        initial_matrix = np.asarray(initial_matrix, dtype=float)
        self.column_count = initial_matrix.shape[1] if initial_matrix.ndim == 2 else 1
        if allow_row_count:
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel('Rows'))
            self.row_spin = QSpinBox()
            self.row_spin.setRange(0, 100)
            self.row_spin.valueChanged.connect(self._resize_rows)
            row_layout.addWidget(self.row_spin)
            row_layout.addStretch()
            self.editor_layout.addLayout(row_layout)

        self.table = QTableWidget()
        self.table.setItemDelegate(FloatDelegate())
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumSize(120, 65)
        self.editor_layout.addWidget(self.table)
        self.set_matrix(initial_matrix)

    def _resize_rows(self, row_count):
        """Resize the table while preserving existing entries where possible."""
        saved_rows = []
        for row in range(self.table.rowCount()):
            saved_rows.append([
                self.table.item(row, column).text()
                if self.table.item(row, column) is not None else '0.0'
                for column in range(self.column_count)
            ])
        self.table.setRowCount(row_count)
        self.table.setColumnCount(self.column_count)
        for row in range(row_count):
            for column in range(self.column_count):
                text = (
                    saved_rows[row][column]
                    if row < len(saved_rows) else '0.0'
                )
                self.table.setItem(row, column, QTableWidgetItem(text))

    def set_matrix(self, matrix):
        """Populate the editor with a matrix of the configured column count."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.column_count:
            raise ValueError('Matrix dimensions do not match the configured editor.')
        if self.allow_row_count:
            self.row_spin.blockSignals(True)
            self.row_spin.setValue(matrix.shape[0])
            self.row_spin.blockSignals(False)
        self.table.setRowCount(matrix.shape[0])
        self.table.setColumnCount(self.column_count)
        for row in range(matrix.shape[0]):
            for column in range(self.column_count):
                self.table.setItem(
                    row, column, QTableWidgetItem(f'{matrix[row, column]:.8g}')
                )

    def get_matrix(self):
        """Read the table and reject empty, non-finite, or invalid cells."""
        matrix = np.zeros((self.table.rowCount(), self.column_count), dtype=float)
        for row in range(self.table.rowCount()):
            for column in range(self.column_count):
                item = self.table.item(row, column)
                if item is None or not item.text().strip():
                    raise ValueError('Matrix entries cannot be empty.')
                try:
                    matrix[row, column] = float(item.text())
                except ValueError as error:
                    raise ValueError('Matrix entries must be valid numbers.') from error
        if not np.all(np.isfinite(matrix)):
            raise ValueError('Matrix entries must be finite numbers.')
        return matrix


class MPCIndexedSettingsDialog(QDialog):
    """Edit cost weights or constraints for one selected MPC stage."""

    def __init__(self, parent, horizon, state_count, values, settings_type):
        super().__init__(parent)
        self.horizon = int(horizon)
        self.state_count = int(state_count)
        self.settings_type = settings_type
        self.values = copy.deepcopy(values)
        self._loading = False
        self._current_stage = 0

        self.setWindowTitle(
            'Set optimisation function'
            if settings_type == 'cost' else 'Set optimisation constraints'
        )
        layout = QVBoxLayout(self)
        stage_layout = QHBoxLayout()
        stage_layout.addWidget(QLabel('Stage i'))
        self.stage_combo = QComboBox()
        for stage in range(self.horizon + 1):
            self.stage_combo.addItem(str(stage), stage)
        self.stage_combo.currentIndexChanged.connect(self._stage_changed)
        stage_layout.addWidget(self.stage_combo)
        stage_layout.addStretch()
        layout.addLayout(stage_layout)

        if settings_type == 'cost':
            self.V_xx_editor = NumericMatrixInput(
                'V_xx_i', np.eye(self.state_count), parent=self
            )
            layout.addWidget(self.V_xx_editor)
            weight_layout = QHBoxLayout()
            self.V_yy_edit = self._new_nonnegative_edit()
            self.V_uu_edit = self._new_nonnegative_edit()
            weight_layout.addWidget(QLabel('V_yy_i'))
            weight_layout.addWidget(self.V_yy_edit)
            weight_layout.addWidget(QLabel('V_uu_i'))
            weight_layout.addWidget(self.V_uu_edit)
            layout.addLayout(weight_layout)
        else:
            self.M_editor = NumericMatrixInput(
                'M_i', np.empty((0, self.state_count)),
                allow_row_count=True, parent=self
            )
            self.N_editor = NumericMatrixInput(
                'N_i', np.empty((0, 1)),
                allow_row_count=True, parent=self
            )
            self.b_editor = NumericMatrixInput(
                'b_i', np.empty((0, 1)),
                allow_row_count=True, parent=self
            )
            self.M_editor.row_spin.valueChanged.connect(self._sync_constraint_rows)
            self.N_editor.row_spin.valueChanged.connect(self._sync_constraint_rows)
            self.b_editor.row_spin.valueChanged.connect(self._sync_constraint_rows)
            matrix_layout = QHBoxLayout()
            matrix_layout.addWidget(self.M_editor)
            matrix_layout.addWidget(self.N_editor)
            matrix_layout.addWidget(self.b_editor)
            layout.addLayout(matrix_layout)

        self.whole_horizon_button = QPushButton('Set for whole horizon')
        self.whole_horizon_button.clicked.connect(self._set_for_whole_horizon)
        layout.addWidget(self.whole_horizon_button)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._load_stage(0)

    @staticmethod
    def _new_nonnegative_edit():
        edit = QLineEdit('1.0')
        edit.setValidator(QDoubleValidator(0.0, 1e100, 12, edit))
        return edit

    def _current_value(self):
        if self.settings_type == 'cost':
            V_yy = float(self.V_yy_edit.text())
            V_uu = float(self.V_uu_edit.text())
            return {
                'V_xx': self.V_xx_editor.get_matrix(),
                'V_yy': V_yy,
                'V_uu': V_uu,
            }
        return {
            'M': self.M_editor.get_matrix(),
            'N': self.N_editor.get_matrix(),
            'b': self.b_editor.get_matrix(),
        }

    def _save_current(self):
        self.values[self._current_stage] = self._current_value()

    def _load_stage(self, stage):
        self._loading = True
        self._current_stage = stage
        value = self.values[stage]
        if self.settings_type == 'cost':
            self.V_xx_editor.set_matrix(value['V_xx'])
            self.V_yy_edit.setText(str(value['V_yy']))
            self.V_uu_edit.setText(str(value['V_uu']))
            terminal = stage == self.horizon
            self.V_yy_edit.setEnabled(not terminal)
            self.V_uu_edit.setEnabled(not terminal)
        else:
            self.M_editor.set_matrix(value['M'])
            self.N_editor.set_matrix(value['N'])
            self.b_editor.set_matrix(value['b'])
            self.N_editor.setEnabled(stage != self.horizon)
        self._loading = False

    def _stage_changed(self, index):
        if self._loading:
            return
        next_stage = int(self.stage_combo.itemData(index))
        previous_stage = self._current_stage
        try:
            self._save_current()
        except (ValueError, KeyError) as error:
            QMessageBox.warning(self, 'Invalid settings', str(error))
            self.stage_combo.blockSignals(True)
            self.stage_combo.setCurrentIndex(previous_stage)
            self.stage_combo.blockSignals(False)
            return
        self._load_stage(next_stage)

    def _sync_constraint_rows(self, row_count):
        if self._loading:
            return
        for editor in (self.M_editor, self.N_editor, self.b_editor):
            if editor.row_spin.value() != row_count:
                editor.row_spin.blockSignals(True)
                editor.row_spin.setValue(row_count)
                editor.row_spin.blockSignals(False)
                editor._resize_rows(row_count)

    def _set_for_whole_horizon(self):
        try:
            current_value = self._current_value()
        except (ValueError, KeyError) as error:
            QMessageBox.warning(self, 'Invalid settings', str(error))
            return

        # Terminal stages have no input/output cost or input constraints;
        # preserve those stage-specific values while sharing applicable matrices.
        for stage in range(self.horizon + 1):
            if self.settings_type == 'cost':
                updated = copy.deepcopy(self.values[stage])
                updated['V_xx'] = current_value['V_xx'].copy()
                if self._current_stage < self.horizon and stage < self.horizon:
                    updated['V_yy'] = current_value['V_yy']
                    updated['V_uu'] = current_value['V_uu']
                self.values[stage] = updated
            else:
                updated = copy.deepcopy(self.values[stage])
                updated['M'] = current_value['M'].copy()
                updated['b'] = current_value['b'].copy()
                if self._current_stage < self.horizon and stage < self.horizon:
                    updated['N'] = current_value['N'].copy()
                self.values[stage] = updated
        self._save_current()

    def _accept(self):
        try:
            self._save_current()
        except (ValueError, KeyError) as error:
            QMessageBox.warning(self, 'Invalid settings', str(error))
            return
        self.accept()


class StateSpaceMatrixInput(QWidget):
    """Widget to edit continuous-time state-space and noise matrices A, B, C, D, Q, R.

    Public API:
        get_matrices() -> (A, B, C, D)
    """

    def __init__(
        self,
        parent=None,
        initial_dims: int = PLANT_DEFAULT_PARAMS['dims'],
        show_noise_matrices: bool = True,
        show_integrator_controls: bool = True,
        allow_dimension_change: bool = True,
    ):
        super().__init__(parent)

        self.dims = max(1, int(initial_dims))
        self.use_ode_mode = False
        self.integrator_method = IntegratorType.EULER_MARUYAMA

        self.delegate = FloatDelegate()

        self.spin = QSpinBox()
        self.spin.setMinimum(1)
        self.spin.setValue(self.dims)
        self.spin.valueChanged.connect(self.on_dims_changed)
        self.spin.setEnabled(allow_dimension_change)

        lbl = QLabel('State dimension')
        top_layout = QVBoxLayout()
        header_layout = QGridLayout()
        header_layout.addWidget(lbl, 0, 0)
        header_layout.addWidget(self.spin, 0, 1)
        self.use_ode_mode_button = QPushButton()
        self.use_ode_mode_button.setCheckable(True)
        self.use_ode_mode_button.toggled.connect(self.set_use_ode_mode)
        header_layout.addWidget(self.use_ode_mode_button, 1, 0, 1, 2)
        self.integrator_label = QLabel('Integrator')
        self.integrator_combo = QComboBox()
        self.integrator_combo.currentIndexChanged.connect(self.on_integrator_changed)
        header_layout.addWidget(self.integrator_label, 2, 0)
        header_layout.addWidget(self.integrator_combo, 2, 1)
        top_layout.addLayout(header_layout)

        # matrices: use QTableWidget for each
        self.table_A = QTableWidget()
        self.table_B = QTableWidget()
        self.table_C = QTableWidget()
        self.table_D = QTableWidget()
        self.table_Q = QTableWidget()
        self.table_R = QTableWidget()

        for t in (self.table_A, self.table_B, self.table_C, self.table_D, self.table_Q, self.table_R):
            t.setItemDelegate(self.delegate)
            t.verticalHeader().setVisible(False)
            t.setMinimumSize(160, 80)

        grid = QGridLayout()
        grid.addWidget(QLabel('A'), 0, 0)
        grid.addWidget(self.table_A, 1, 0)
        grid.addWidget(QLabel('B'), 0, 1)
        grid.addWidget(self.table_B, 1, 1)
        grid.addWidget(QLabel('C'), 2, 0)
        grid.addWidget(self.table_C, 3, 0)
        grid.addWidget(QLabel('D'), 2, 1)
        grid.addWidget(self.table_D, 3, 1)
        self.label_Q = QLabel('Q')
        grid.addWidget(self.label_Q, 0, 2)
        grid.addWidget(self.table_Q, 1, 2)
        self.label_R = QLabel('R')
        grid.addWidget(self.label_R, 2, 2)
        grid.addWidget(self.table_R, 3, 2)

        top_layout.addLayout(grid)
        self.setLayout(top_layout)

        self.col_width = 80
        self.resize_all(self.dims)
        self.set_use_ode_mode(False)
        if not show_noise_matrices:
            self.label_Q.hide()
            self.table_Q.hide()
            self.label_R.hide()
            self.table_R.hide()
        if not show_integrator_controls:
            self.use_ode_mode_button.hide()
            self.integrator_label.hide()
            self.integrator_combo.hide()

    def set_use_ode_mode(self, use_ode_mode: bool):
        prev_method = self.integrator_method
        self.use_ode_mode = bool(use_ode_mode)

        if self.use_ode_mode:
            self.label_Q.setText('Q (covariance matrix)')
            self.use_ode_mode_button.setText('use_ode_mode: On')
        else:
            self.label_Q.setText('Q (diffusion matrix)')
            self.use_ode_mode_button.setText('use_ode_mode: Off')

        if self.use_ode_mode_button.isChecked() != self.use_ode_mode:
            self.use_ode_mode_button.blockSignals(True)
            try:
                self.use_ode_mode_button.setChecked(self.use_ode_mode)
            finally:
                self.use_ode_mode_button.blockSignals(False)

        self.populate_integrator_options(preferred_method=prev_method)

    def get_integrator_choices(self) -> list[tuple[str, IntegratorType]]:
        if self.use_ode_mode:
            return [
                ('RK4', IntegratorType.RK4),
                ('Analytic ODE', IntegratorType.ANALYTIC_ODE),
            ]

        return [
            ('Euler-Maruyama', IntegratorType.EULER_MARUYAMA),
            ('Analytic SDE', IntegratorType.ANALYTIC_SDE),
        ]

    def normalize_integrator_method(self, method: IntegratorType | None) -> IntegratorType:
        available = [m for _, m in self.get_integrator_choices()]
        if method in available:
            return method

        if self.use_ode_mode:
            if method == IntegratorType.ANALYTIC_SDE:
                return IntegratorType.ANALYTIC_ODE
            return IntegratorType.RK4

        if method == IntegratorType.ANALYTIC_ODE:
            return IntegratorType.ANALYTIC_SDE
        return IntegratorType.EULER_MARUYAMA

    def populate_integrator_options(self, preferred_method: IntegratorType | None = None):
        choices = self.get_integrator_choices()
        selected = self.normalize_integrator_method(preferred_method)

        self.integrator_combo.blockSignals(True)
        try:
            self.integrator_combo.clear()
            for text, method in choices:
                self.integrator_combo.addItem(text, method)

            index = 0
            for i in range(self.integrator_combo.count()):
                if self.integrator_combo.itemData(i) == selected:
                    index = i
                    break
            self.integrator_combo.setCurrentIndex(index)
        finally:
            self.integrator_combo.blockSignals(False)

        self.integrator_method = self.integrator_combo.currentData()

    def set_integrator_method(self, method: IntegratorType):
        self.populate_integrator_options(preferred_method=method)

    def get_integrator_method(self) -> IntegratorType:
        method = self.integrator_combo.currentData()
        if method is None:
            raise ValueError('No integrator selected.')
        return method

    def on_integrator_changed(self, _index: int):
        method = self.integrator_combo.currentData()
        if method is not None:
            self.integrator_method = method

    def on_dims_changed(self, val: int):
        val = max(1, int(val))
        self.dims = val
        self.resize_all(val)

    def fill_table_with_zeros(self, table: QTableWidget, rows: int, cols: int):
        table.clearContents()
        table.setRowCount(rows)
        table.setColumnCount(cols)
        for c in range(cols):
            table.setColumnWidth(c, self.col_width)
        for r in range(rows):
            for c in range(cols):
                item = QTableWidgetItem('0.0')
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)

    def resize_all(self, dims: int):
        self.fill_table_with_zeros(self.table_A, dims, dims)
        self.fill_table_with_zeros(self.table_B, dims, 1)
        self.fill_table_with_zeros(self.table_C, 1, dims)
        self.fill_table_with_zeros(self.table_D, 1, 1)
        self.fill_table_with_zeros(self.table_Q, dims, dims)
        self.fill_table_with_zeros(self.table_R, 1, 1)

    def set_table_ABCD(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        """Set the table contents from numpy arrays. Shapes must match dims.

        This will resize the internal tables to match the provided `A` shape.
        """
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('A must be square')
        dims = int(A.shape[0])

        if B.shape != (dims, 1) or C.shape != (1, dims) or D.shape != (1, 1):
            raise ValueError('Matrix shapes do not match')

        # set spin to trigger resizing
        self.spin.blockSignals(True)
        try:
            self.spin.setValue(dims)
            self.dims = dims
            self.resize_all(dims)
        finally:
            self.spin.blockSignals(False)

        # fill tables
        for i in range(dims):
            for j in range(dims):
                self.table_A.item(i, j).setText(f'{float(A[i, j]):.6g}')

        for i in range(dims):
            self.table_B.item(i, 0).setText(f'{float(B[i, 0]):.6g}')

        for j in range(dims):
            self.table_C.item(0, j).setText(f'{float(C[0, j]):.6g}')

        self.table_D.item(0, 0).setText(f'{float(D[0, 0]):.6g}')

    def set_table_QR(self, Q: np.ndarray, R: np.ndarray):
        Q = np.asarray(Q)
        R = np.asarray(R)
        if Q.shape != (self.dims, self.dims) or R.shape != (1, 1):
            raise ValueError('Noise matrix shapes do not match')

        for i in range(self.dims):
            for j in range(self.dims):
                self.table_Q.item(i, j).setText(f'{float(Q[i, j]):.6g}')
        self.table_R.item(0, 0).setText(f'{float(R[0, 0]):.6g}')

    def get_table_ABCD(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (A, B, C, D) as numpy arrays of shapes (dims, dims), (dims, 1), (1, dims), (1, 1).
        Raises ValueError if any cell is empty or contains invalid float.
        """
        try:
            A = np.zeros((self.dims, self.dims), dtype=float)
            for i in range(self.dims):
                for j in range(self.dims):
                    cell_val = self.table_A.item(i, j)
                    if cell_val is None or cell_val.text().strip() == '':
                        raise ValueError('Matrix contains empty or invalid entries')
                    A[i, j] = float(cell_val.text())

            B = np.zeros((self.dims, 1), dtype=float)
            for i in range(self.dims):
                cell_val = self.table_B.item(i, 0)
                if cell_val is None or cell_val.text().strip() == '':
                    raise ValueError('Matrix contains empty or invalid entries')
                B[i, 0] = float(cell_val.text())

            C = np.zeros((1, self.dims), dtype=float)
            for j in range(self.dims):
                cell_val = self.table_C.item(0, j)
                if cell_val is None or cell_val.text().strip() == '':
                    raise ValueError('Matrix contains empty or invalid entries')
                C[0, j] = float(cell_val.text())

            cell_val = self.table_D.item(0, 0)
            if cell_val is None or cell_val.text().strip() == '':
                raise ValueError('Matrix contains empty or invalid entries')
            D = np.array([[float(cell_val.text())]], dtype=float)

            return A, B, C, D
        except ValueError:
            raise
        except Exception:
            raise ValueError('Matrix contains empty or invalid entries')

    def get_table_QR(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            Q = np.zeros((self.dims, self.dims), dtype=float)
            for i in range(self.dims):
                for j in range(self.dims):
                    cell_val = self.table_Q.item(i, j)
                    if cell_val is None or cell_val.text().strip() == '':
                        raise ValueError('Matrix contains empty or invalid entries')
                    Q[i, j] = float(cell_val.text())

            cell_val = self.table_R.item(0, 0)
            if cell_val is None or cell_val.text().strip() == '':
                raise ValueError('Matrix contains empty or invalid entries')
            R = np.array([[float(cell_val.text())]], dtype=float)
            return Q, R
        except ValueError:
            raise
        except Exception:
            raise ValueError('Matrix contains empty or invalid entries')


class FloatDelegate(QStyledItemDelegate):
    """Item delegate that restricts editing in Qt cells to floating-point numbers."""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        editor.setValidator(validator)
        editor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return editor
