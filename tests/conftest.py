# pragma: no cover
if __name__ == "__main__":
    import __init__  # noqa

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from main import Simulation


@pytest.fixture(scope="session")
def app():
    '''
    Create a QApplication once per test session.
    Qt cleans up on exit, no manual closure needed.
    '''
    app = QApplication(sys.argv)
    yield app

@pytest.fixture
def sim(app):
    '''
    Provide a fresh Simulation object for each test.
    '''
    return Simulation()
