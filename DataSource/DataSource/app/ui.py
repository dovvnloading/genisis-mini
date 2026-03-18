import html
import json

import ollama
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import (
    APP_STYLESHEET,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_VALIDATION_MODEL,
    LOG_COLOR_MAP,
    PREFERRED_QWEN_MODELS,
)
from .utils import get_next_filename
from .workers import DataGenerationWorker, TopicExpansionWorker


def create_app_icon():
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#007acc"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(pixmap.rect(), 8, 8)
    painter.setPen(QColor("white"))
    font = painter.font()
    font.setFamily("Segoe UI")
    font.setPixelSize(44)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "D")
    painter.end()
    return QIcon(pixmap)


class SyntheticDatasetGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genisis-Mini")
        self.setWindowIcon(create_app_icon())
        self.setGeometry(100, 100, 800, 700)
        self.client = ollama.Client(timeout=45)
        self.critical_errors = []
        self.setStyleSheet(APP_STYLESHEET)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(12, 20, 12, 12)
        controls_layout.setSpacing(12)
        form_layout = QFormLayout()
        form_layout.setHorizontalSpacing(15)
        form_layout.setVerticalSpacing(10)

        self.main_topic_edit = QLineEdit("The history and impact of the Internet")
        self.target_topics_spin = QSpinBox()
        self.target_topics_spin.setRange(10, 5000)
        self.target_topics_spin.setValue(50)
        self.total_samples_spin = QSpinBox()
        self.total_samples_spin.setRange(1, 10000)
        self.total_samples_spin.setValue(100)
        self.similarity_threshold_spin = QDoubleSpinBox()
        self.similarity_threshold_spin.setRange(0.7, 1.0)
        self.similarity_threshold_spin.setValue(0.95)
        self.similarity_threshold_spin.setSingleStep(0.01)
        self.similarity_threshold_spin.setDecimals(2)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 50)
        self.batch_size_spin.setValue(7)
        self.generation_model_combo = QComboBox()
        self.validation_model_combo = QComboBox()
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["JSONL (.jsonl)", "JSON Array (.json)"])
        self.populate_model_combos()

        form_layout.addRow("Main Topic:", self.main_topic_edit)
        form_layout.addRow("Target Unique Topics:", self.target_topics_spin)
        form_layout.addRow("Total Samples:", self.total_samples_spin)
        form_layout.addRow("Similarity Threshold:", self.similarity_threshold_spin)
        form_layout.addRow("Generation Batch Size:", self.batch_size_spin)
        form_layout.addRow("Generation Model:", self.generation_model_combo)
        form_layout.addRow("Validation Model:", self.validation_model_combo)
        form_layout.addRow("Export Format:", self.export_format_combo)

        controls_layout.addLayout(form_layout)
        self.generate_btn = QPushButton("Start Generation Process")
        self.generate_btn.clicked.connect(self.start_generation_process)
        controls_layout.addWidget(self.generate_btn)
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        self.log_group = QGroupBox()
        self.log_group.setProperty("isError", False)
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(12, 12, 12, 12)
        self.log_group.setLayout(log_layout)

        log_header_layout = QHBoxLayout()
        log_header_layout.setContentsMargins(0, 0, 0, 0)
        log_title_label = QLabel("Process Log")
        log_title_label.setStyleSheet("font-weight: bold; color: #888; font-size: 12px; padding: 0 8px;")

        self.error_flag_btn = QPushButton("View Errors")
        self.error_flag_btn.setStyleSheet("background-color: #c0392b; padding: 4px 8px; font-size: 11px;")
        self.error_flag_btn.setToolTip("Click to view critical errors that occurred during the process.")
        self.error_flag_btn.clicked.connect(self.toggle_error_display)
        self.error_flag_btn.hide()

        log_header_layout.addWidget(log_title_label)
        log_header_layout.addSpacerItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        log_header_layout.addWidget(self.error_flag_btn)
        log_layout.addLayout(log_header_layout)

        self.error_display_widget = QWidget()
        error_display_layout = QVBoxLayout(self.error_display_widget)
        error_display_layout.setContentsMargins(2, 8, 2, 2)
        self.error_text_edit = QTextEdit()
        self.error_text_edit.setReadOnly(True)
        self.error_text_edit.setMaximumHeight(120)
        self.copy_error_btn = QPushButton("Copy Error Text")
        self.copy_error_btn.setStyleSheet("padding: 6px; font-size: 11px;")
        self.copy_error_btn.clicked.connect(self.copy_error_text)
        error_display_layout.addWidget(self.error_text_edit)
        error_display_layout.addWidget(self.copy_error_btn)
        self.error_display_widget.hide()
        log_layout.addWidget(self.error_display_widget)

        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        log_layout.addWidget(self.output_edit)
        main_layout.addWidget(self.log_group, 1)

    def populate_model_combos(self):
        model_names = list(PREFERRED_QWEN_MODELS)
        installed_model_names = []

        try:
            response = self.client.list()
            for model in response.get("models", []):
                name = model.get("model") or model.get("name")
                if name:
                    installed_model_names.append(name)
        except Exception:
            installed_model_names = []

        for name in installed_model_names:
            if name not in model_names:
                model_names.append(name)

        self.generation_model_combo.addItems(model_names)
        self.validation_model_combo.addItems(model_names)
        self.set_combo_to_text(self.generation_model_combo, DEFAULT_GENERATION_MODEL)
        self.set_combo_to_text(self.validation_model_combo, DEFAULT_VALIDATION_MODEL)

    @staticmethod
    def set_combo_to_text(combo_box, target_text):
        index = combo_box.findText(target_text)
        if index >= 0:
            combo_box.setCurrentIndex(index)

    def log_message(self, message_tuple):
        message, level = message_tuple
        color = LOG_COLOR_MAP.get(level, "#cccccc")
        safe_message = html.escape(message)
        html_message = (
            f"<hr><h4 style='color: {color};'>&raquo; {safe_message}</h4>"
            if level == "STAGE"
            else f"<p style='color: {color}; margin: 2px 0;'>{safe_message}</p>"
        )
        self.output_edit.append(html_message)
        scroll_bar = self.output_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def start_generation_process(self):
        main_topic = self.main_topic_edit.text().strip()
        if not main_topic:
            self.log_message(("Input Error: Please provide a Main Topic.", "ERROR"))
            return

        self.generate_btn.setEnabled(False)
        self.output_edit.clear()
        self.critical_errors.clear()
        self.error_flag_btn.hide()
        self.error_display_widget.hide()
        self.error_text_edit.clear()
        self.log_group.setProperty("isError", False)
        self.log_group.style().polish(self.log_group)

        self.total_samples = self.total_samples_spin.value()
        self.generation_model = self.generation_model_combo.currentText()
        self.validation_model = self.validation_model_combo.currentText()
        self.log_message((f"Using generation model '{self.generation_model}' and validation model '{self.validation_model}'.", "INFO"))

        self.topic_worker = TopicExpansionWorker(
            main_topic,
            self.target_topics_spin.value(),
            self.similarity_threshold_spin.value(),
            self.batch_size_spin.value(),
            self.generation_model,
            self.validation_model,
            self.client,
        )
        self.topic_worker.progress.connect(self.log_message)
        self.topic_worker.finished.connect(self.on_topics_expanded)
        self.topic_worker.error.connect(self.generation_error)
        self.topic_worker.start()

    def on_topics_expanded(self, sub_topics):
        if not sub_topics:
            self.generation_error("Topic expansion resulted in an empty list. Cannot proceed.")
            return

        self.log_message((f"Topic Expansion Complete. Final unique topic count: {len(sub_topics)}", "SUCCESS"))
        self.log_message(("STAGE 2: DATASET GENERATION & VALIDATION", "STAGE"))

        self.data_worker = DataGenerationWorker(
            sub_topics,
            self.total_samples,
            self.generation_model,
            self.validation_model,
            self.client,
        )
        self.data_worker.progress.connect(self.log_message)
        self.data_worker.finished.connect(self.generation_finished)
        self.data_worker.error.connect(self.generation_error)
        self.data_worker.start()

    def generation_finished(self, dataset):
        self.log_message(("PROCESS COMPLETE", "STAGE"))
        if not dataset:
            self.log_message(("Final dataset is empty. This may be due to repeated generation/validation failures.", "WARN"))
            self.generate_btn.setEnabled(True)
            return

        export_format = self.export_format_combo.currentText()
        extension = ".jsonl" if "JSONL" in export_format else ".json"
        filename = get_next_filename(extension=extension)

        try:
            with open(filename, "w", encoding="utf-8") as file_handle:
                if extension == ".jsonl":
                    for item in dataset:
                        file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    json.dump(dataset, file_handle, ensure_ascii=False, indent=4)
            self.log_message((f"Dataset successfully saved to {filename} ({len(dataset)} valid samples).", "SUCCESS"))
        except Exception as exc:
            self.generation_error(f"Error saving file: {exc}")
            return

        self.generate_btn.setEnabled(True)

    def generation_error(self, error_msg):
        self.log_message((f"FATAL ERROR: {error_msg}", "ERROR"))
        self.critical_errors.append(error_msg)
        self.log_group.setProperty("isError", True)
        self.log_group.style().polish(self.log_group)

        error_count = len(self.critical_errors)
        plural = "s" if error_count > 1 else ""
        self.error_flag_btn.setText(f"View {error_count} Critical Error{plural}!")
        self.error_flag_btn.show()

        formatted_errors = "\n\n".join(f"- {error}" for error in self.critical_errors)
        self.error_text_edit.setText(formatted_errors)
        self.generate_btn.setEnabled(True)

    def toggle_error_display(self):
        self.error_display_widget.setVisible(not self.error_display_widget.isVisible())

    def copy_error_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.error_text_edit.toPlainText())
        self.log_message(("Error text copied to clipboard.", "INFO"))
