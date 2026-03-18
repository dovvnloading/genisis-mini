import sys

from PySide6.QtWidgets import QApplication

from app.ui import SyntheticDatasetGenerator


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SyntheticDatasetGenerator()
    window.show()
    sys.exit(app.exec())
