from enum import Enum, auto


class PlotType(Enum):
    HIDE = auto()
    BODE = auto()
    NYQUIST = auto()
    NICHOLS = auto()
    ROOTLOCUS = auto()
    
    def __str__(self):
        """Return the string representation for display purposes"""
        return self.name
