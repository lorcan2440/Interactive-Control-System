# built-ins
import logging
import sys
from datetime import datetime

# external imports
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QHBoxLayout, QLabel, QWidget


#########################
### Editable Settings ###
#########################

# set to True to log to console and debug.log
LOGGING_ON = True

# default slider parameters for controllers and plant
GUI_SLIDER_CONFIG = {
    'manual_u':      {'min': -5.0,  'max': 5.0,     'step': 0.1,    'init': 0.0},
    'y_sp':          {'min': -1.0,  'max': 1.0,     'step': 0.01,   'init': 1.0},
    'w_proc_stddev': {'min': 0.0,   'max': 2.0,     'step': 0.01,   'init': 0.0},
    'w_meas_stddev': {'min': 0.0,   'max': 0.5,     'step': 0.01,   'init': 0.0},
    'U_plus':        {'min': 0.0,   'max': 5.0,     'step': 0.1,    'init': 1.0},
    'U_minus':       {'min': -5.0,  'max': 0.0,     'step': 0.1,    'init': -1.0},
    'K_p':           {'min': 0.0,   'max': 200.0,   'step': 1.0,    'init': 10.0},
    'K_i':           {'min': 0.0,    'max': 20.0,    'step': 0.05,   'init': 0.0},
    'K_d':           {'min': 0.0,    'max': 50.0,    'step': 0.05,   'init': 0.0},
}

# time step sizes for integration, animation and sliding window
# it is better (but not required) to keep these as integer multiples of each other
TIME_STEPS = {
    'DT_INT': 0.001,
    'DT_ANIM': 0.050,
    'DT_SLIDING_WINDOW': 3.000,
}

# ratio of real time to simulation time for the animation (e.g. 1.0 = real time, 2.0 = double speed, 0.5 = half speed)
# NOTE: graphics rendering or computations may still be limiting factors
ANIM_SPEED_FACTOR = 1.0

#########################
### Internal settings ###
#########################

# for testing equality of floats
MAX_SIG_FIGS = 10

# controller parameters available in the GUI
CONTROLLER_PARAMS_LIST = ['manual_u', 'K_p', 'K_i', 'K_d', 'U_plus', 'U_minus']

# default process and measurement noise variances
w_proc_var_init = GUI_SLIDER_CONFIG['w_proc_stddev']['init'] ** 2
w_meas_var_init = GUI_SLIDER_CONFIG['w_meas_stddev']['init'] ** 2

# default plant state space model parameters and initial conditions
PLANT_DEFAULT_PARAMS = {
     
    'A': np.array([[-11.0, 20.0],       # state transition matrix A (from x to x_dot)
                   [10.0, -21.0]]),

	'B': np.array([[1.0],               # control input matrix B (from u to x_dot)
                   [0.0]]),

	'C': np.array([[0.0, 1.0]]),        # measurement matrix C (from x to y)
     
	'D': np.array([[0.0]]),             # feedthrough matrix D (from u to y)
     
	'Q': np.array([[w_proc_var_init, 0.0],      # process noise covariance matrix Q (from w_proc to x_dot)
                   [0.0, w_proc_var_init]]),

	'R': np.array([[w_meas_var_init]]),         # measurement noise covariance matrix R (from w_meas to y)
     
    'x_0': np.array([[1.0],             # initial state x_0
                     [1.0]]),
}

# if D is missing, set it to [[0.0]]
if 'D' not in PLANT_DEFAULT_PARAMS:
    PLANT_DEFAULT_PARAMS['D'] = np.array([[0.0]])

# if D is non-zero, we need to set an initial control input u_0: default to zero unless u_0 already set above
if PLANT_DEFAULT_PARAMS['D'].item() != 0.0 and 'u_0' not in PLANT_DEFAULT_PARAMS:
    PLANT_DEFAULT_PARAMS['u_0'] = np.array([[0.0]])

#########################
### Utility functions ###
#########################

class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        if not datefmt:
            return super().formatTime(record, datefmt=datefmt)

        return datetime.fromtimestamp(record.created).astimezone().strftime(datefmt)


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a configured logger. This function is idempotent - calling it
    multiple times with the same `name` will not add duplicate handlers.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers if logger already configured

    formatter = MicrosecondFormatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt="%Y-%m-%d %H:%M:%S.%f")

    file_handler = logging.FileHandler('debug.log')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def make_slider_from_cfg(key: str, display_name: str = None, 
        orientation: Qt.Orientation = Qt.Orientation.Horizontal) -> \
            tuple[QWidget, QSlider, QLabel]:
    """
    Create a slider row from a config key.

    This function requires `key` to be a string present in
    `GUI_SLIDER_CONFIG` and always returns a tuple
    `(container_widget, slider, value_label)`.

    Args:
        key: the key into `GUI_SLIDER_CONFIG`.
        display_name: optional label text to show left of the slider.
        orientation: slider orientation.

    Returns:
        `(QWidget, QSlider, QLabel)` representing the row container, the
        slider, and the value label.
    """

    if key not in GUI_SLIDER_CONFIG:
        raise KeyError(f"Unknown slider key: {key}")

    cfg = GUI_SLIDER_CONFIG[key]
    if cfg['step'] <= 0:
        raise ValueError(f"Slider step must be > 0 for '{key}' (got {cfg['step']})")

    # calculate number of steps
    n_steps = max(1, int(round((cfg['max'] - cfg['min']) / cfg['step'])))

    slider = QSlider(orientation)
    slider.setMinimum(0)
    slider.setMaximum(n_steps)

    # calculate initial integer position
    init_int = int(round((cfg.get('init', cfg['min']) - cfg['min']) / cfg['step']))
    init_int = min(max(init_int, 0), n_steps)
    slider.setValue(init_int)
    slider.setSingleStep(1)
    slider.setPageStep(max(1, n_steps // 10))

    # build full row widget
    container = QWidget()
    row = QHBoxLayout()
    label = QLabel(display_name or key)
    row.addWidget(label)
    row.addWidget(slider)
    val_label = QLabel(f"{cfg.get('init', cfg['min']):.2f}")
    row.addWidget(val_label)
    container.setLayout(row)

    return container, slider, val_label
