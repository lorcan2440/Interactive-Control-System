from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QGroupBox, QRadioButton, QCheckBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from controllers import ControllerType


class GUI:
    def __init__(self, simulator):
        self.simulator = simulator
    
    def user_interface(self):
        '''
        Contains the sliders and plots on screen.
        '''

        layout = QVBoxLayout()  # overall layout: vertical

        #### Plots ####

        self.simulator.fig, (self.simulator.ax1, self.simulator.ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
        self.simulator.canvas = FigureCanvas(self.simulator.fig)
        layout.addWidget(self.simulator.canvas)

        # Initialize plot data for both true and measured outputs
        self.simulator.t_data = [0]
        self.simulator.y_true_data = [self.simulator.state_x[1]]  # true output y = x2
        self.simulator.y_measured_data = [self.simulator.state_x[1]]  # measured output (initially same)
        self.simulator.u_data = [self.simulator.last_u]

        # Output plot with both true and measured signals
        self.simulator.ax1.set_ylabel("Outputs")
        self.simulator.line_y_true, = self.simulator.ax1.plot(
            self.simulator.t_data, self.simulator.y_true_data, 'b-', label="$ x_2 $")
        self.simulator.line_y_measured, = self.simulator.ax1.plot(
            self.simulator.t_data, self.simulator.y_measured_data, 'kx', markersize=3, label="$ y = x_2 + w_2 $")
        self.simulator.ref_line, = self.simulator.ax1.plot(self.simulator.t_data, [self.simulator.setpoint], 'r--', label="Setpoint")
        self.simulator.ax1.legend(loc="upper right")
        self.simulator.ax1.set_xlim(0, 5)
        self.simulator.ax1.set_ylim(-2, 2)

        # Control input plot
        self.simulator.ax2.set_xlabel("Time")
        self.simulator.ax2.set_ylabel("Input")
        self.simulator.line_u, = self.simulator.ax2.plot(self.simulator.t_data, self.simulator.u_data, 'g-', label="$ u $")
        self.simulator.ax2.legend(loc="upper right")
        self.simulator.ax2.set_xlim(0, 5)
        self.simulator.ax2.set_ylim(-5, 5)

        #### Simulation Settings ####

        control_layout = QVBoxLayout()

        # controller selection radio buttons
        controller_select_box = QGroupBox("Controller Selection")
        controller_select_layout = QHBoxLayout()
        
        self.simulator.manual_radio = QRadioButton("Manual Control")
        self.simulator.bangbang_radio = QRadioButton("Bang-Bang Control")
        self.simulator.openloop_radio = QRadioButton("Open Loop Control")
        self.simulator.pid_radio = QRadioButton("PID Control")
        self.simulator.h2_radio = QRadioButton("H2 Optimal Control (LQG)")

        self.simulator.manual_radio.toggled.connect(self.simulator.on_controller_changed)
        self.simulator.openloop_radio.toggled.connect(self.simulator.on_controller_changed)
        self.simulator.bangbang_radio.toggled.connect(self.simulator.on_controller_changed)
        self.simulator.pid_radio.toggled.connect(self.simulator.on_controller_changed)
        self.simulator.h2_radio.toggled.connect(self.simulator.on_controller_changed)
        
        controller_select_layout.addWidget(self.simulator.manual_radio)
        controller_select_layout.addWidget(self.simulator.openloop_radio)
        controller_select_layout.addWidget(self.simulator.bangbang_radio)
        controller_select_layout.addWidget(self.simulator.pid_radio)
        controller_select_layout.addWidget(self.simulator.h2_radio)

        controller_select_box.setLayout(controller_select_layout)
        control_layout.addWidget(controller_select_box)

        # horizontal layout for setpoint and noise boxes
        setpoint_noise_layout = QHBoxLayout()
        
        # setpoint control (left side)
        setpoint_box = QGroupBox("Setpoint (reference)")
        setpoint_layout = QVBoxLayout()

        self.simulator.setpoint_slider = self.simulator.make_slider_from_cfg('setpoint')
        self.simulator.setpoint_slider.valueChanged.connect(self.simulator.update_setpoint)
        self.simulator.setpoint_label = QLabel(f"Setpoint: {self.simulator.setpoint:.2f}")
        
        setpoint_layout.addWidget(self.simulator.setpoint_slider)
        setpoint_layout.addWidget(self.simulator.setpoint_label)
        setpoint_box.setLayout(setpoint_layout)
        setpoint_noise_layout.addWidget(setpoint_box)

        # noise control box (right side)
        noise_box = QGroupBox("Disturbances")
        noise_layout = QGridLayout()
        
        # w1_stddev (process noise)
        self.simulator.w1_stddev_slider = self.simulator.make_slider_from_cfg('w1_stddev')
        self.simulator.w1_stddev_slider.valueChanged.connect(self.simulator.update_w1_stddev)
        self.simulator.w1_stddev_label = QLabel(f"Process Noise (w1): {self.simulator.w1_stddev:.3f}")
        noise_layout.addWidget(self.simulator.w1_stddev_slider, 0, 0)
        noise_layout.addWidget(self.simulator.w1_stddev_label, 1, 0)
        
        # w2_stddev (measurement noise)
        self.simulator.w2_stddev_slider = self.simulator.make_slider_from_cfg('w2_stddev')
        self.simulator.w2_stddev_slider.valueChanged.connect(self.simulator.update_w2_stddev)
        self.simulator.w2_stddev_label = QLabel(f"Measurement Noise (w2): {self.simulator.w2_stddev:.3f}")
        noise_layout.addWidget(self.simulator.w2_stddev_slider, 0, 1)
        noise_layout.addWidget(self.simulator.w2_stddev_label, 1, 1)
        
        noise_box.setLayout(noise_layout)
        setpoint_noise_layout.addWidget(noise_box)
        
        # add the horizontal layout to the main control layout
        control_layout.addLayout(setpoint_noise_layout)

        #### Control Parameters ####

        ### Manual Controls ###
        self.simulator.manual_box = QGroupBox("Manual Controller Parameters")
        manual_layout = QVBoxLayout()
        
        # Enable/disable checkbox
        self.simulator.manual_enable_checkbox = QCheckBox("Enable Manual Control")
        self.simulator.manual_enable_checkbox.setChecked(self.simulator.manual_enabled)
        self.simulator.manual_enable_checkbox.stateChanged.connect(self.simulator.update_manual_enabled)
        manual_layout.addWidget(self.simulator.manual_enable_checkbox)
        
        # Manual control slider
        self.simulator.manual_slider = self.simulator.make_slider_from_cfg('manual_u')
        self.simulator.manual_slider.valueChanged.connect(self.simulator.update_manual_u)
        self.simulator.manual_label = QLabel(f"Manual Control u: {self.simulator.manual_u:.2f}")
        manual_layout.addWidget(self.simulator.manual_slider)
        manual_layout.addWidget(self.simulator.manual_label)
        
        self.simulator.manual_box.setLayout(manual_layout)
        control_layout.addWidget(self.simulator.manual_box)

        ### Open Loop Controls ###
        # None

        ### Bang Bang Controls ###
        self.simulator.bangbang_box = QGroupBox("Bang Bang Controller Parameters")
        bangbang_layout = QGridLayout()

        # U_minus
        self.simulator.U_minus_slider = self.simulator.make_slider_from_cfg('U_minus')
        self.simulator.U_minus_slider.valueChanged.connect(self.simulator.update_U_minus)
        self.simulator.U_minus_label = QLabel(f"U_minus: {self.simulator.U_minus:.2f}")
        bangbang_layout.addWidget(self.simulator.U_minus_slider, 0, 0)
        bangbang_layout.addWidget(self.simulator.U_minus_label, 1, 0)
    
        # U_plus
        self.simulator.U_plus_slider = self.simulator.make_slider_from_cfg('U_plus')
        self.simulator.U_plus_slider.valueChanged.connect(self.simulator.update_U_plus)
        self.simulator.U_plus_label = QLabel(f"U_plus: {self.simulator.U_plus:.2f}")
        bangbang_layout.addWidget(self.simulator.U_plus_slider, 0, 1)
        bangbang_layout.addWidget(self.simulator.U_plus_label, 1, 1)

        self.simulator.bangbang_box.setLayout(bangbang_layout)
        control_layout.addWidget(self.simulator.bangbang_box)

        ### PID Controls ###
        self.simulator.pid_box = QGroupBox("PID Controller Parameters")
        pid_layout = QGridLayout()

        # Kp
        self.simulator.Kp_slider = self.simulator.make_slider_from_cfg('Kp')
        self.simulator.Kp_slider.valueChanged.connect(self.simulator.update_Kp)
        self.simulator.Kp_label = QLabel(f"Kp: {self.simulator.Kp:.2f}")
        pid_layout.addWidget(self.simulator.Kp_slider, 0, 0)
        pid_layout.addWidget(self.simulator.Kp_label, 1, 0)

        # Ki
        self.simulator.Ki_slider = self.simulator.make_slider_from_cfg('Ki')
        self.simulator.Ki_slider.valueChanged.connect(self.simulator.update_Ki)
        self.simulator.Ki_label = QLabel(f"Ki: {self.simulator.Ki:.2f}")
        pid_layout.addWidget(self.simulator.Ki_slider, 0, 1)
        pid_layout.addWidget(self.simulator.Ki_label, 1, 1)

        # Kd
        self.simulator.Kd_slider = self.simulator.make_slider_from_cfg('Kd')
        self.simulator.Kd_slider.valueChanged.connect(self.simulator.update_Kd)
        self.simulator.Kd_label = QLabel(f"Kd: {self.simulator.Kd:.2f}")
        pid_layout.addWidget(self.simulator.Kd_slider, 0, 2)
        pid_layout.addWidget(self.simulator.Kd_label, 1, 2)

        self.simulator.pid_box.setLayout(pid_layout)
        control_layout.addWidget(self.simulator.pid_box)

        ### H2 Controls ###
        self.simulator.h2_box = QGroupBox("H2 Controller Parameters")
        h2_layout = QGridLayout()
        
        # C1_1
        self.simulator.h2_C1_1_slider = self.simulator.make_slider_from_cfg('h2_C1_1')
        self.simulator.h2_C1_1_slider.valueChanged.connect(self.simulator.update_h2_C1_1)
        self.simulator.h2_C1_1_label = QLabel(f"C1_1 (performance gain of x_1): {self.simulator.h2_C1_1:.2f}")
        h2_layout.addWidget(self.simulator.h2_C1_1_slider, 0, 0)
        h2_layout.addWidget(self.simulator.h2_C1_1_label, 1, 0)
        
        # C1_2
        self.simulator.h2_C1_2_slider = self.simulator.make_slider_from_cfg('h2_C1_2')
        self.simulator.h2_C1_2_slider.valueChanged.connect(self.simulator.update_h2_C1_2)
        self.simulator.h2_C1_2_label = QLabel(f"C1_2 (performance gain of x_2): {self.simulator.h2_C1_2:.2f}")
        h2_layout.addWidget(self.simulator.h2_C1_2_slider, 0, 1)
        h2_layout.addWidget(self.simulator.h2_C1_2_label, 1, 1)
        
        self.simulator.h2_box.setLayout(h2_layout)
        control_layout.addWidget(self.simulator.h2_box)

        ### Set starting object states

        self.simulator.manual_box.setVisible(True)  # the manual box is shown first, others hidden
        self.simulator.h2_box.setVisible(False)
        self.simulator.pid_box.setVisible(False)

        # set controller radio button type
        match self.simulator.controller_type:
            case ControllerType.MANUAL:
                self.simulator.manual_radio.setChecked(True)
            case ControllerType.OPENLOOP:
                self.simulator.openloop_radio.setChecked(True)
            case ControllerType.BANGBANG:
                self.simulator.bangbang_radio.setChecked(True)
            case ControllerType.PID:
                self.simulator.pid_radio.setChecked(True)
            case ControllerType.H2:
                self.simulator.h2_radio.setChecked(True)

        # set manual slider enabled state based on checkbox
        self.simulator.update_manual_slider_state()

        layout.addLayout(control_layout)
        self.simulator.setLayout(layout)