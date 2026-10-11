# Interactive Control System

<video src="https://raw.githubusercontent.com/lorcan2440/Interactive-Control-System/main/media/Recording%202026-10-11%20102742.mp4" controls width="100%">
  <a href="https://github.com/lorcan2440/Interactive-Control-System/blob/main/media/Recording%202026-10-11%20102742.mp4">Watch the demo video</a>
</video>

## Currently available controllers

- **Manual** - set the control input yourself, using your mouse
- **Open-loop** - no feedback, just calculate the control input based on setpoint and plant gain.
- **Bang-bang** - positive input if output is too low, negative input if output is too high
- **PID** - the classic industry-standard, including option of a filtered derivative on measurement to help with practical stability.
- **H2 optimal control** - aka linear quadratic regulator (LQR) with a Kalman filter or linear quadratic Gaussian (LQG). Minimises the energy gain from the noise disturbance input to the performance output (z = C1_x @ x + C1_u * u, where C1_x and C1_u are customisable gains).
- **H-infinity optimal control** - minimises the worst-case gain between disturbance and performance output across all frequencies.
- **Model predictive control (MPC)** - forecasts the future states for a given control input trajectory and solves a constrained quadratic least-squares problem to minimise the error and control effort. The relative weights and constraints are customisable.
- **Reinforcement learning (RL) control** - an 'actor' learns to map observed measurements to continuous control inputs (policy), and a 'critic' learns to evaluate their cost (value function), based on a reward signal that penalises large errors, large values of input and large changes input, both represented by neural networks. Three such actor-critic RL algorithms are available: deep deterministic policy gradient (DDPG), twin-delayed DDPG (TD3) and soft actor-critic (SAC). Neural net architecture and algorithm parameters are customisable within.

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
