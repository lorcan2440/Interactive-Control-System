# Interactive Control System

https://github.com/user-attachments/assets/9ebc596b-dc7c-4065-8524-7f06ba27772c

## Requirements

External libraries are NumPy (matrices), SciPy (linear algebra and integration), Matplotlib (plotting) and PyQt6 (GUI). Install using:

`$ pip install numpy scipy matplotlib PyQt6`

Alternatively, `PyQt6` can be swapped out for `PySide6` by replacing the `import` lines with:

```python
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QSlider, QLabel, QGroupBox, QRadioButton)
from PySide6.QtCore import Qt, QTimer
```

Does latex work here $ \mathcal{H}_{\infty} (G(s)) = 1.0 $

## Future ideas

- [ ] Plot the error signal e instead of the output y in the top subplot.
- [ ] Add feedforward and bang-bang control.
- [ ] Show the poles of the OLTF $ L(s) $ in the complex plane and allow interactive pole placement.
- [ ] Show a Bode/Nyquist plot of $ L(s) $ with the gain/phase margins and allow switching between them.
- [ ] Add a lead-lag compensator with interactive loop-shaping
- [ ] Add a $ H_∞ $ optimal controller, either by solving the CAREs or the LMI using CVX.
- [ ] Add an MPC using OSQP with editable objective function, constraints and horizon (at this point we may need to rethink the UI as it would be getting cluttered - only show buttons/sliders for the controller being used)
- [ ] Add an RL-based controller like DDPG (probably way too much to fit inside this project, would need a new program, could maybe borrow from stable_baselines)
- [ ] Rewrite to run in a browser (no idea how to do this at present... JavaScript? 😭
