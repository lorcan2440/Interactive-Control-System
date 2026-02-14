# Interactive Control System

https://github.com/user-attachments/assets/4b876526-36ec-4d46-a481-f202f1f5bdc2

## Currently available controllers

- [x] Manual control (choose the control input with your mouse!)
- [x] Feedforward control (aka open loop control)
- [x] Bang-bang control (aka on/off control)
- [x] PID control
- [ ] H2 optimal control (aka LQG control)

## Requirements

The libraries required are NumPy, SciPy, PyQt6 and PyQtGraph. 
You can install the most up-to-date versions using:

```bash
pip install numpy scipy PyQt6 pyqtgraph
```

or see below for installing specific versions for guaranteed compatability.

`PyQt6` can be easily swapped out for `PySide6` by replacing the `import` lines in `gui.py` and `main.py` with:

```python
from PySide6.QtWidgets import ...
from PySide6.QtCore import ...
```

## How to run

1. Clone the repo to your system:

```bash
git clone https://github.com/lorcan2440/Interactive-Control-System.git
```

2. Navigate into the project directory:

```bash
cd Interactive-Control-System
```

(Optional) Create and activate a virtual environment. This helps keep dependencies isolated.

```bash
python -m venv venv
# if on Linux / macOS:
source venv/bin/activate
# in on Windows:
venv\Scripts\activate
```

3. Make sure you have Python 3 with pip installed, then run:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python main.py
```

## Future controllers to be added at some point

- [ ] Lead-lag compensator
- [ ] Sliding mode control (including boundary layer smoothing) - may replace bang-bang
- [ ] Model predictive control (solve using OSQP)
- [ ] H2 optimal control (aka LQG control)
- [ ] H∞ optimal control (solve by either CARE or LMI in CVX)
- [ ] Neural control (using LSTM)
- [ ] Reinforcement learning control (using DDPG)

## Other future to-dos

- [ ] Plot the error signal e instead of the output y in the top subplot.
- [ ] Show the poles of the OLTF L(s) in the complex plane and allow interactive pole placement.
- [ ] Show a Bode/Nyquist plot of L(s) with the gain/phase margins and allow switching between them.
- [ ] Add interactive loop-shaping to the lead-lag compensator
