# external imports
import numpy as np
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QRadioButton, QPushButton, QButtonGroup
from PyQt6.QtCore import Qt
from pyqtgraph import GraphicsLayoutWidget, mkPen

# local imports
from controllers import ControllerType
from utils import make_slider_from_cfg, GUI_SLIDER_CONFIG, CONTROLLER_PARAMS_LIST


class GUI:

    def __init__(self, sim: object):

        self.sim = sim
        self.logger = self.sim.logger

        # plotting buffers
        self.t_data = np.array([])
        self.x1 = np.array([])
        self.x2 = np.array([])
        self.y_meas = np.array([])
        self.y_sp_data = np.array([])
        self.u_data = np.array([])

        # controller UI bookkeeping
        self.sim.controller_type = ControllerType.MANUAL
        self.controller_param_widgets = {}

    def init_gui(self):

        # build layout onto the Simulation QWidget
        main_layout = QVBoxLayout()

        ## 1) top area - graphs

        self.win = GraphicsLayoutWidget()  # from PyQtGraph

        # states and measurement plot
        self.plot_x = self.win.addPlot(row=0, col=0, title='States (x)')
        self.plot_x.addLegend()
        self.curve_x1 = self.plot_x.plot(pen='r', name='x1')
        self.curve_x2 = self.plot_x.plot(pen='b', name='x2')
        self.curve_x1.setVisible(False)
        self.curve_y_meas = self.plot_x.plot(pen=mkPen('y', width=1), name='y_meas')
        self.curve_y_sp = self.plot_x.plot(pen=mkPen('m', style=Qt.PenStyle.DashLine), name='y_sp')

        # control input plot
        self.plot_u = self.win.addPlot(row=1, col=0, title='Control input (u)')
        self.curve_u = self.plot_u.plot(pen='g')

        main_layout.addWidget(self.win, stretch=1)

        ## 2) first row under graphs - controller selection

        # start/stop button
        first_row_hbox = QHBoxLayout()
        self.start_stop_button = QPushButton('Start')
        self.start_stop_button.clicked.connect(self.toggle_start_stop)
        first_row_hbox.addWidget(self.start_stop_button)
        first_row_hbox.addStretch()

        # controller selection box
        controller_buttons_box = QGroupBox('Controller Selection')
        controller_buttons_box_layout = QHBoxLayout()
        self.controller_buttons_group = QButtonGroup()

        # controller selection radio buttons
        self.radio_manual = QRadioButton('Manual Control')
        self.radio_openloop = QRadioButton('Open Loop Control')
        self.radio_bangbang = QRadioButton('Bang-Bang Control')
        self.radio_pid = QRadioButton('PID Control')

        self.controller_buttons_group.addButton(self.radio_manual)
        self.controller_buttons_group.addButton(self.radio_openloop)
        self.controller_buttons_group.addButton(self.radio_bangbang)
        self.controller_buttons_group.addButton(self.radio_pid)

        controller_buttons_box_layout.addWidget(self.radio_manual)
        controller_buttons_box_layout.addWidget(self.radio_openloop)
        controller_buttons_box_layout.addWidget(self.radio_bangbang)
        controller_buttons_box_layout.addWidget(self.radio_pid)

        controller_buttons_box.setLayout(controller_buttons_box_layout)
        first_row_hbox.addWidget(controller_buttons_box)
        main_layout.addLayout(first_row_hbox)

        ## 3) second row under graphs - setpoint and disturbances
    
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

        # disturbances
        dist_box = QGroupBox('Disturbances')
        dist_layout = QHBoxLayout()
        w1_cfg = GUI_SLIDER_CONFIG['w_proc_stddev']
        w2_cfg = GUI_SLIDER_CONFIG['w_meas_stddev']

        w1_box = QVBoxLayout()
        w1_container, w1_slider, w1_val_label = make_slider_from_cfg('w_proc_stddev', 'Process Noise (w_proc)')
        w1_slider.valueChanged.connect(self.on_w_slider_changed)
        w1_val_label.setText(f'{w1_cfg.get("init", w1_cfg["min"]):.3f}')
        w1_box.addWidget(w1_container)
        self.slider_w1 = w1_slider
        self.w1_label = w1_val_label

        w2_box = QVBoxLayout()
        w2_container, w2_slider, w2_val_label = make_slider_from_cfg('w_meas_stddev', 'Measurement Noise (w_meas)')
        w2_slider.valueChanged.connect(self.on_w_slider_changed)
        w2_val_label.setText(f'{w2_cfg.get("init", w2_cfg["min"]):.3f}')
        w2_box.addWidget(w2_container)
        self.slider_w2 = w2_slider
        self.w2_label = w2_val_label

        dist_layout.addLayout(w1_box)
        dist_layout.addLayout(w2_box)
        dist_box.setLayout(dist_layout)
        second_hbox.addWidget(dist_box, stretch=4)

        main_layout.addLayout(second_hbox)

        ## 4) third row under graphs - controller parameters (dynamically generated)
        self.params_box = QGroupBox('Controller Parameters')
        self.params_layout = QHBoxLayout()
        self.params_box.setLayout(self.params_layout)
        main_layout.addWidget(self.params_box)

        self.sim.setLayout(main_layout)

        # connect controller radio buttons
        self.radio_manual.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.MANUAL))
        self.radio_openloop.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.OPENLOOP))
        self.radio_bangbang.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.BANGBANG))
        self.radio_pid.toggled.connect(lambda on: on and self.on_controller_selected(ControllerType.PID))

        # initial controller selection and params
        self.radio_manual.setChecked(True)
        self.sim.manual_u = GUI_SLIDER_CONFIG['manual_u']['init']
        self.set_controller(ControllerType.MANUAL)
        self.build_controller_params(ControllerType.MANUAL)

    def update_plots(self, t_span: np.ndarray, x_span: np.ndarray, y_meas: np.ndarray):
        # t_span: array of times for this frame. Shape: (n,)
        # x_span: state trajectory for this frame. Shape: (2, n)
        # y_meas: latest measurement. Shape: (1, 1)

        # compute setpoint numeric value
        sp_cfg = GUI_SLIDER_CONFIG['y_sp']
        y_sp_val = sp_cfg['min'] + int(self.slider.value()) * sp_cfg['step']

        # u used during this frame (constant across this frame)
        u_last = float(self.sim.plant.u.item())

        # current measured output (scalar)
        y_last = float(y_meas.item())

        if self.t_data.size == 0:
            # first frame: take full time span and states
            u_0, y_0 = self.sim.plant.u_0.item(), self.sim.y_0.item()
            self.t_data = t_span.copy()  # shape: (n,)
            self.x1 = x_span[0, :].copy()  # shape: (n,)
            self.x2 = x_span[1, :].copy()  # shape: (n,)
            self.t_meas = np.array([float(t_span[0]), float(t_span[-1])])  # shape: (2,)
            self.y_meas = np.array([y_0, y_last])  # shape: (2,)
            self.u_data = np.full(self.t_data.shape, u_0)  # shape: (n,)
            self.y_sp_data = np.full(self.t_data.shape, y_sp_val)  # shape: (n,)
        else:
            # subsequent frames: append times and state trajectory for this frame 
            # (skip first sample to avoid duplication)
            self.t_data = np.hstack((self.t_data, t_span[1:]))
            self.x1 = np.hstack((self.x1, x_span[0, 1:]))
            self.x2 = np.hstack((self.x2, x_span[1, 1:]))
            self.t_meas = np.hstack((self.t_meas, float(t_span[-1])))
            self.y_meas = np.hstack((self.y_meas, y_last))
            self.u_data = np.hstack((self.u_data, np.full(t_span.shape[0] - 1, u_last)))
            self.y_sp_data = np.hstack((self.y_sp_data, np.full(t_span.shape[0] - 1, y_sp_val)))
        
        # apply sliding window: truncate arrays
        # retain only entries where mask is True (in the window)
        if self.t_data.size > 0:
            mask_data_in_window = (self.t_data >= self.t_data[-1] - self.sim.dt_window)
            mask_meas_in_window = (self.t_meas >= self.t_meas[-1] - self.sim.dt_window)
            self.t_data = self.t_data[mask_data_in_window]
            self.t_meas = self.t_meas[mask_meas_in_window]
            self.x1 = self.x1[mask_data_in_window]
            self.x2 = self.x2[mask_data_in_window]
            self.u_data = self.u_data[mask_data_in_window]
            self.y_sp_data = self.y_sp_data[mask_data_in_window]
            self.y_meas = self.y_meas[mask_meas_in_window]

        # update curves
        self.curve_x1.setData(self.t_data, self.x1)
        self.curve_x2.setData(self.t_data, self.x2)
        self.curve_y_meas.setData(self.t_meas, self.y_meas)
        self.curve_y_sp.setData(self.t_data, self.y_sp_data)
        self.curve_u.setData(self.t_data, self.u_data)

    ## UI callbacks

    def toggle_start_stop(self):
        # toggle the simulation timer on and off
        if not self.sim.running:  # start
            self.sim.timer.start(int(self.sim.dt_anim * 1000))
            self.sim.running = True
            self.start_stop_button.setText('Stop')
        else:  # stop
            self.sim.timer.stop()
            self.sim.running = False
            self.start_stop_button.setText('Start')

    def on_setpoint_slider_changed(self, val: int):
        # update setpoint based on slider value
        cfg = GUI_SLIDER_CONFIG['y_sp']
        sp = cfg['min'] + val * cfg['step']
        self.sp_label.setText(f'{sp:.2f}')
        self.sim.y_sp = np.array([[sp]])

    def on_w_slider_changed(self, _val: int):
        # update both plant noise covariances

        # w1: process noise stddev
        cfg_w1 = GUI_SLIDER_CONFIG['w_proc_stddev']
        w1_sigma = cfg_w1['min'] + int(self.slider_w1.value()) * cfg_w1['step']
        self.w1_label.setText(f'{w1_sigma:.3f}')

        # w2: measurement noise stddev
        cfg_w2 = GUI_SLIDER_CONFIG['w_meas_stddev']
        w2_sigma = cfg_w2['min'] + int(self.slider_w2.value()) * cfg_w2['step']
        self.w2_label.setText(f'{w2_sigma:.3f}')

        Q = np.array([[w1_sigma**2, 0.0], [0.0, w1_sigma**2]])
        R = np.array([[w2_sigma**2]])

        self.sim.plant.set_noise_covariances(Q=Q, R=R)

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
        # set the simulation controller type
        match controller_type:
            case ControllerType.MANUAL:
                self.sim.controller_type = ControllerType.MANUAL
            case ControllerType.BANGBANG:
                self.sim.controller_type = ControllerType.BANGBANG
            case ControllerType.OPENLOOP:
                self.sim.controller_type = ControllerType.OPENLOOP
            case ControllerType.PID:
                self.sim.controller_type = ControllerType.PID
                self.sim.pid_controller.error_integrated = np.array([[0.0]])