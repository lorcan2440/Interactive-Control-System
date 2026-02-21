# external imports
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QRadioButton, QPushButton, \
    QButtonGroup, QDialog, QDialogButtonBox, QMessageBox, QWidget, QGridLayout, QSpinBox, QTableWidget, \
    QTableWidgetItem, QStyledItemDelegate, QLineEdit
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDoubleValidator
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, mkPen

# local imports
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
        self.radio_manual = QRadioButton('Manual Control')
        self.radio_openloop = QRadioButton('Open Loop Control')
        self.radio_bangbang = QRadioButton('Bang-Bang Control')
        self.radio_pid = QRadioButton('PID Control')

        self.controller_buttons_group.addButton(self.radio_none)
        self.controller_buttons_group.addButton(self.radio_manual)
        self.controller_buttons_group.addButton(self.radio_openloop)
        self.controller_buttons_group.addButton(self.radio_bangbang)
        self.controller_buttons_group.addButton(self.radio_pid)

        controller_buttons_box_layout.addWidget(self.radio_none)
        controller_buttons_box_layout.addWidget(self.radio_manual)
        controller_buttons_box_layout.addWidget(self.radio_openloop)
        controller_buttons_box_layout.addWidget(self.radio_bangbang)
        controller_buttons_box_layout.addWidget(self.radio_pid)

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

        # show descriptive text in the value label for consistency
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

    ## UI callbacks

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

    def add_param(self, key: str, display_name: str = None):
        # helper: create a controller parameter slider row
        container, slider, val_label = make_slider_from_cfg(key, display_name)
        slider.valueChanged.connect(lambda v, k=key: self.on_controller_param_changed(k, v))
        self.params_layout.addWidget(container)
        self.controller_param_widgets[key] = (slider, val_label)

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

        # set slider positions to current values
        for key, (slider, val_label) in self.controller_param_widgets.items():
            cfg = GUI_SLIDER_CONFIG[key]
            current_val = float(getattr(self.sim, key, cfg.get('init', cfg['min'])))
            pos = int(round((current_val - cfg['min']) / cfg['step']))
            pos = min(max(pos, 0), int(round((cfg['max'] - cfg['min']) / cfg['step'])))
            slider.setValue(pos)
            val_label.setText(f"{current_val:.2f}")

    def on_controller_param_changed(self, key: str, int_pos: int):
        cfg = GUI_SLIDER_CONFIG[key]
        if cfg is None:
            return
        val = cfg['min'] + int_pos * cfg['step']
        _, val_label = self.controller_param_widgets.get(key, (None, None))
        if val_label is not None:
            val_label.setText(f"{val:.2f}")

        if key in CONTROLLER_PARAMS_LIST:
            setattr(self.sim, key, val)
        
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

            # pre-fill with current plant matrices if available
            A, B, C, D = self.sim.plant.A, self.sim.plant.B, self.sim.plant.C, self.sim.plant.D
            widget.set_matrices(np.asarray(A), np.asarray(B), np.asarray(C), np.asarray(D))
            Q, R = self.sim.plant.Q, self.sim.plant.R
            widget.set_noise_matrices(np.asarray(Q), np.asarray(R))

            dlg_layout.addWidget(widget)

            eig_label = QLabel('Eigenvalues:')
            dlg_layout.addWidget(eig_label)

            def _update_eigs(*_):
                try:
                    A_cur, _, _, _ = widget.get_matrices()
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

        A_new, B_new, C_new, D_new = widget.get_matrices()
        Q_new, R_new = widget.get_noise_matrices()

        try:
            # apply new matrices to plant
            self.sim.plant.set_state_space_matrices(A_new, B_new, C_new, D_new)
            self.sim.plant.set_noise_covariances(Q=Q_new, R=R_new)

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

            if str(e) in ("Noise covariance matrices Q and R have incorrect dimensions.", 
                          "Process noise covariance matrix Q must be symmetric.",
                          "Noise covariance matrices Q and R must be positive semidefinite."):
                QMessageBox.warning(dialog, 'Invalid noise covariance matrices', str(e))
                return

            # apply the new state-space matrices
            self.sim.plant.set_state_space_matrices(A_new, B_new, C_new, D_new)
            self.sim.plant.set_noise_covariances(Q=Q_new, R=R_new)

            # clear data buffers and graph areas
            self.clear_buffers()
            self.clear_graph_traces()

        # for controllers that use plant matrices, recalculate any of their needed params
        self.sim.openloop_controller.calc_ss_gain()
        self.sim.pid_controller.reset_memory()

        dialog.accept()


class StateSpaceMatrixInput(QWidget):
    """Widget to edit continuous-time state-space and noise matrices A, B, C, D, Q, R.

    Public API:
        get_matrices() -> (A, B, C, D)
    """

    def __init__(self, parent=None, initial_dims: int = PLANT_DEFAULT_PARAMS['dims']):
        super().__init__(parent)

        self.dims = max(1, int(initial_dims))

        self.delegate = FloatDelegate()

        self.spin = QSpinBox()
        self.spin.setMinimum(1)
        self.spin.setValue(self.dims)
        self.spin.valueChanged.connect(self.on_dims_changed)

        lbl = QLabel('State dimension')
        top_layout = QVBoxLayout()
        header_layout = QGridLayout()
        header_layout.addWidget(lbl, 0, 0)
        header_layout.addWidget(self.spin, 0, 1)
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
        grid.addWidget(QLabel('Q'), 0, 2)
        grid.addWidget(self.table_Q, 1, 2)
        grid.addWidget(QLabel('R'), 2, 2)
        grid.addWidget(self.table_R, 3, 2)

        top_layout.addLayout(grid)
        self.setLayout(top_layout)

        self.col_width = 80
        self.resize_all(self.dims)

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

    def set_matrices(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
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

    def set_noise_matrices(self, Q: np.ndarray, R: np.ndarray):
        Q = np.asarray(Q)
        R = np.asarray(R)
        if Q.shape != (self.dims, self.dims) or R.shape != (1, 1):
            raise ValueError('Noise matrix shapes do not match')

        for i in range(self.dims):
            for j in range(self.dims):
                self.table_Q.item(i, j).setText(f'{float(Q[i, j]):.6g}')
        self.table_R.item(0, 0).setText(f'{float(R[0, 0]):.6g}')

    def get_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def get_noise_matrices(self) -> tuple[np.ndarray, np.ndarray]:
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
