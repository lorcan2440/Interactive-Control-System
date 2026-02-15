import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QStyledItemDelegate,
    QLineEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator

from utils import PLANT_DEFAULT_PARAMS


class _FloatDelegate(QStyledItemDelegate):
    """Item delegate that restricts editing to floating-point numbers."""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        editor.setValidator(validator)
        editor.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return editor


class StateSpaceMatrixInput(QWidget):
    """Widget to edit continuous-time state-space matrices A, B, C, D.

    Public API:
        get_matrices() -> (A, B, C, D)
    """

    def __init__(self, parent=None, initial_dims: int = PLANT_DEFAULT_PARAMS['dims']):
        super().__init__(parent)

        self._dims = max(1, int(initial_dims))

        self._delegate = _FloatDelegate()

        self._spin = QSpinBox()
        self._spin.setMinimum(1)
        self._spin.setValue(self._dims)
        self._spin.valueChanged.connect(self._on_dims_changed)

        lbl = QLabel('State dimension')
        top_layout = QVBoxLayout()
        header_layout = QGridLayout()
        header_layout.addWidget(lbl, 0, 0)
        header_layout.addWidget(self._spin, 0, 1)
        top_layout.addLayout(header_layout)

        # matrices: use QTableWidget for each
        self._table_A = QTableWidget()
        self._table_B = QTableWidget()
        self._table_C = QTableWidget()
        self._table_D = QTableWidget()

        for t in (self._table_A, self._table_B, self._table_C, self._table_D):
            t.setItemDelegate(self._delegate)
            t.verticalHeader().setVisible(False)
            t.setMinimumSize(160, 80)

        grid = QGridLayout()
        grid.addWidget(QLabel('A'), 0, 0)
        grid.addWidget(self._table_A, 1, 0)
        grid.addWidget(QLabel('B'), 0, 1)
        grid.addWidget(self._table_B, 1, 1)
        grid.addWidget(QLabel('C'), 2, 0)
        grid.addWidget(self._table_C, 3, 0)
        grid.addWidget(QLabel('D'), 2, 1)
        grid.addWidget(self._table_D, 3, 1)

        top_layout.addLayout(grid)
        self.setLayout(top_layout)

        self._col_width = 80
        self._resize_all(self._dims)

    def _on_dims_changed(self, val: int):
        val = max(1, int(val))
        self._dims = val
        self._resize_all(val)

    def _fill_table_with_zeros(self, table: QTableWidget, rows: int, cols: int):
        table.clearContents()
        table.setRowCount(rows)
        table.setColumnCount(cols)
        for c in range(cols):
            table.setColumnWidth(c, self._col_width)
        for r in range(rows):
            for c in range(cols):
                item = QTableWidgetItem('0.0')
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(r, c, item)

    def _resize_all(self, dims: int):
        self._fill_table_with_zeros(self._table_A, dims, dims)
        self._fill_table_with_zeros(self._table_B, dims, 1)
        self._fill_table_with_zeros(self._table_C, 1, dims)
        self._fill_table_with_zeros(self._table_D, 1, 1)

    def set_matrices(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        """Set the table contents from numpy arrays. Shapes must match dims.

        This will resize the internal tables to match the provided `A` shape.
        """
        A = np.asarray(A)
        B = np.asarray(B)
        C = np.asarray(C)
        D = np.asarray(D)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('A must be square')
        dims = int(A.shape[0])

        if B.shape != (dims, 1) or C.shape != (1, dims) or D.shape != (1, 1):
            raise ValueError('Matrix shapes do not match')

        # set spin to trigger resizing
        self._spin.blockSignals(True)
        try:
            self._spin.setValue(dims)
            self._dims = dims
            self._resize_all(dims)
        finally:
            self._spin.blockSignals(False)

        # fill tables
        for i in range(dims):
            for j in range(dims):
                self._table_A.item(i, j).setText(f'{float(A[i, j]):.6g}')

        for i in range(dims):
            self._table_B.item(i, 0).setText(f'{float(B[i, 0]):.6g}')

        for j in range(dims):
            self._table_C.item(0, j).setText(f'{float(C[0, j]):.6g}')

        self._table_D.item(0, 0).setText(f'{float(D[0, 0]):.6g}')

    def get_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (A, B, C, D) as numpy arrays of shapes (dims, dims), (dims, 1), (1, dims), (1, 1).
        Raises ValueError if any cell is empty or contains invalid float.
        """
        try:
            A = np.zeros((self._dims, self._dims), dtype=float)
            for i in range(self._dims):
                for j in range(self._dims):
                    cell_val = self._table_A.item(i, j)
                    if cell_val is None or cell_val.text().strip() == '':
                        raise ValueError('Matrix contains empty or invalid entries')
                    A[i, j] = float(cell_val.text())

            B = np.zeros((self._dims, 1), dtype=float)
            for i in range(self._dims):
                cell_val = self._table_B.item(i, 0)
                if cell_val is None or cell_val.text().strip() == '':
                    raise ValueError('Matrix contains empty or invalid entries')
                B[i, 0] = float(cell_val.text())

            C = np.zeros((1, self._dims), dtype=float)
            for j in range(self._dims):
                cell_val = self._table_C.item(0, j)
                if cell_val is None or cell_val.text().strip() == '':
                    raise ValueError('Matrix contains empty or invalid entries')
                C[0, j] = float(cell_val.text())

            cell_val = self._table_D.item(0, 0)
            if cell_val is None or cell_val.text().strip() == '':
                raise ValueError('Matrix contains empty or invalid entries')
            D = np.array([[float(cell_val.text())]], dtype=float)

            return A, B, C, D
        except ValueError:
            raise
        except Exception:
            raise ValueError('Matrix contains empty or invalid entries')
