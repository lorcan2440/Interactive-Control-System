# built-ins
import logging
import sys
from datetime import datetime

# external imports
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider, QHBoxLayout, QLabel, QWidget


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        if not datefmt:
            return super().formatTime(record, datefmt=datefmt)

        return datetime.fromtimestamp(record.created).astimezone().strftime(datefmt)


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

TIME_STEPS = {
    'DT_SLIDING_WINDOW': 5.0,
    'DT_ANIM': 0.02,
    'DT_INT': 0.001
}

CONTROLLER_PARAMS_LIST = ['manual_u', 'K_p', 'K_i', 'K_d', 'U_plus', 'U_minus']

w_proc_var_init = GUI_SLIDER_CONFIG['w_proc_stddev']['init'] ** 2
w_meas_var_init = GUI_SLIDER_CONFIG['w_meas_stddev']['init'] ** 2

PLANT_DEFAULT_PARAMS = {
     
    'A': np.array([[-11.0, 20.0], 
                   [10.0, -21.0]]),

	'B': np.array([[1.0], 
                   [0.0]]),

	'C': np.array([[0.0, 1.0]]),
     
	'D': np.array([[0.0]]),
     
	'Q': np.array([[w_proc_var_init, 0.0], 
                   [0.0, w_proc_var_init]]),

	'R': np.array([[w_meas_var_init]]),
     
    'x_0': np.array([[1.0], 
                     [1.0]]),
                     
    'u_0': np.array([[0.0]]),
}


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a configured logger. This function is idempotent — calling it
    multiple times with the same `name` will not add duplicate handlers.
    """

    logger = logging.getLogger(name)
    # If the logger already has handlers configured, assume it's been set up
    # and return it as-is to avoid duplicate messages.
    if logger.handlers:
        return logger

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
