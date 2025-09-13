# Interactive Control System

https://github.com/user-attachments/assets/9ebc596b-dc7c-4065-8524-7f06ba27772c

## Requirements

The only libraries used are NumPy (matrices), SciPy (linear algebra and integration), Matplotlib (plotting) and PyQt6 (GUI). You can install the most up-to-date versions using:

```bash
pip install numpy scipy matplotlib PyQt6
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

## Future ideas

- [ ] Plot the error signal e instead of the output y in the top subplot.
- [ ] Add feedforward and bang-bang control.
- [ ] Show the poles of the OLTF L(s) in the complex plane and allow interactive pole placement.
- [ ] Show a Bode/Nyquist plot of L(s) with the gain/phase margins and allow switching between them.
- [ ] Add a lead-lag compensator with interactive loop-shaping
- [ ] Add a H_∞ optimal controller, either by solving the CAREs or the LMI using CVX.
- [ ] Add an MPC using OSQP with editable objective function, constraints and horizon (at this point we may need to rethink the UI as it would be getting cluttered - only show buttons/sliders for the controller being used)
- [ ] Add an RL-based controller like DDPG (probably way too much to fit inside this project, would need a new program, could maybe borrow from stable_baselines)
- [ ] Rewrite to run in a browser (no idea how to do this at present... JavaScript? 😭
