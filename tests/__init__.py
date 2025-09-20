# pragma: no cover

"""
Adds source files to path to allow access without installing package.
"""

from os.path import dirname, realpath
import sys

sys.path.append(dirname(dirname(realpath(__file__))))
print(dirname(dirname(realpath(__file__))))