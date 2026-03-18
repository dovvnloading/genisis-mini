EMBEDDING_MODEL_NAME = "nomic-embed-text"
DEFAULT_GENERATION_MODEL = "qwen3:4b"
DEFAULT_VALIDATION_MODEL = "qwen3:1.7b"
PREFERRED_QWEN_MODELS = [
    "qwen3:4b",
    "qwen3:1.7b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:30b",
    "qwen2.5-coder:7b",
]
MODEL_REQUEST_OPTIONS = {"temperature": 0.2}
VALIDATION_REQUEST_OPTIONS = {"temperature": 0.0}
TOPIC_GENERATION_OPTIONS = {"temperature": 0.7}
MIN_SNIPPET_LENGTH = 20
LOG_COLOR_MAP = {
    "INFO": "#cccccc",
    "SUCCESS": "#2ECC71",
    "ERROR": "#E74C3C",
    "WARN": "#F39C12",
    "STAGE": "#3498DB",
    "VALIDATION": "#95a5a6",
}
APP_STYLESHEET = """
    QWidget { background-color: #2b2b2b; color: #e0e0e0; font-family: Segoe UI; font-size: 13px; }
    QGroupBox { font-weight: bold; color: #888; border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 10px; }
    QGroupBox[isError="true"] { border: 1px solid #c0392b; }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #3c3f41; border: 1px solid #555;
        border-radius: 4px; padding: 6px; color: #e0e0e0;
    }
    QSpinBox, QDoubleSpinBox { padding-right: 20px; }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox::down-arrow { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #e0e0e0; margin-right: 5px; }
    QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border; width: 18px; border-left: 1px solid #555; background-color: #3c3f41;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover { background-color: #4a4d4f; }
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #4a4d4f; }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 5px solid #e0e0e0;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #e0e0e0;
    }
    QPushButton { background-color: #007acc; border: none; border-radius: 4px; padding: 10px; color: white; font-weight: bold; }
    QPushButton:hover { background-color: #008ae6; }
    QPushButton:pressed { background-color: #006bb3; border-top: 2px solid #005a99; }
    QPushButton:disabled { background-color: #555; color: #888; }
    QTextEdit { background-color: #212121; border: 1px solid #3c3c3c; border-radius: 4px; padding: 8px; color: #dcdcdc; font-family: Consolas, monaco, monospace; font-size: 14px; }
    QScrollBar:vertical { border: none; background: #3c3f41; width: 12px; margin: 0; }
    QScrollBar::handle:vertical { background: #666; min-height: 20px; border-radius: 6px; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #3c3f41; }
    QToolTip { background-color: #3c3f41; color: #e0e0e0; border: 1px solid #555; padding: 5px; font-size: 12px; }
"""
