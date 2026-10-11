# Interactive Control System


## Currently available controllers

- [x] **Manual control** (choose the control input with your mouse!)
- [x] **Feedforward control** (aka open loop control)
- [x] **Bang-bang control** (aka on/off control)
- [x] **PID control**, including filtered derivative on measurement
- [x] **H2 optimal control (aka LQG control)**, solved via Riccati equations
- [x] **H-infinity** with output feedback control, solved via Riccati equations*
- [x] **Model predictive control** (MPC) with Kalman filter, solved via OSQP
- [x] **Reinforcement Learning** (RL) controllers: three actor-critic algos (DDPG, TD3 and SAC), pretrained models provided and can train new models with custom hyperparameters

## Requirements

The libraries required are:

- NumPy: arrays and state space matrix algebra.
- SciPy: sparse arrays, computing matrix exponentials, solving Riccati equations and working with transfer functions.
- OSQP: solving the quadratic programme (QP) in the MPC algorithm.
- PyTorch: neural networks representing the actor (policy) and critic (value function) in the RL algorithms.
- PyQt6: the GUI.
- PyQtGraph: plotting of graphs compatible with PyQt6.

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

## Future features

### Currently working on:

- [ ] Root locus plots for the PID controller
- [ ] Bode, Nyquist and Nichols plots of the OLTF L(s) with gain/phase margins

### Hopefully coming soon:

- [ ] Auto-tuning of PID controllers (Ziegler-Nichols, Cohen-Coon, IAE/ITAE optimality)
- [ ] Anti-windup control for the integrator in PID
- [ ] Interactive pole placement controller (including lead lag compensators)
- [ ] Interactive Bode plot (loop shaping controllers)

### Maybe coming eventually:

- [ ] Sliding mode control (including boundary layer smoothing) - may replace bang-bang
- [ ] H-infinity synthesis via LMIs - I tried solving H-infinity using linear matrix inequalities (LMIs) using the CVXPY solver but couldn't figure out how to get the right matrix form, and the calculation took a long time. The Riccati equation approach works fine for now, and is the default method of MATLAB's `hinfsyn` anyway.
- [ ] H-infinity loop shaping?
- [ ] Neural control (MPC using LSTM for forecasting, using autograd to get gradients for optimisation? see [Meng *et al.*, 2023](https://www.mdpi.com/2227-9717/11/4/1080))
