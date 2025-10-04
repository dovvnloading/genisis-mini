import sys
import json
import re
import time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFormLayout,
                               QLabel, QLineEdit, QSpinBox, QPushButton, QTextEdit,
                               QMessageBox, QGroupBox, QDoubleSpinBox, QComboBox,
                               QHBoxLayout, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
import ollama

# --- Utility Functions ---

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

def get_next_filename(base_name="synthetic_dataset", extension=".jsonl"):
    output_dir = Path(".")
    base_path = output_dir / base_name
    initial_path = base_path.with_suffix(extension)
    if not initial_path.exists():
        return str(initial_path)
    
    counter = 1
    while True:
        versioned_path = output_dir / f"{base_name}_{counter:03d}{extension}"
        if not versioned_path.exists():
            return str(versioned_path)
        counter += 1

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def extract_json_from_response(content):
    # First, try to find a markdown-style JSON block
    match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        return match.group(1)
    
    # If not found, fall back to the original greedy search
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        return match.group(0)

    return None

# --- Agentic Components ---

class TopicRelevanceValidator:
    def __init__(self, model_name, progress_signal, client):
        self.model_name = model_name
        self.progress_signal = progress_signal
        self.client = client

    def is_relevant(self, main_topic, sub_topic_candidate):
        try:
            prompt = f"""
            You are a relevance validation AI.
            Main Topic: "{main_topic}"
            Sub-Topic Candidate: "{sub_topic_candidate}"
            Is the sub-topic candidate highly relevant to the main topic?
            Your response MUST be a single, raw JSON object and nothing else.
            Example: {{"is_relevant": true}}
            """
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            json_str = extract_json_from_response(response['message']['content'])
            if not json_str: raise ValueError("No JSON object found in relevance validator response.")
            data = json.loads(json_str)
            return data.get("is_relevant", False)
        except Exception as e:
            self.progress_signal.emit((f"   - Relevance validation failed for '{sub_topic_candidate}': {e}. Skipping.", "WARN"))
            return False

class TextSnippetValidator:
    def __init__(self, model_name, progress_signal, client):
        self.model_name = model_name
        self.progress_signal = progress_signal
        self.client = client

    def validate_relevance(self, topic, generated_text):
        self.progress_signal.emit((f"   - Validating snippet relevance for topic: '{topic}'...", "VALIDATION"))
        try:
            prompt = f"""
            You are a Validation AI.
            Topic: "{topic}"
            Text Snippet: "{generated_text}"
            Is the text snippet highly relevant and directly about the topic?
            Your response MUST be a single, raw JSON object and nothing else.
            Example: {{"is_relevant": true, "reason": "The text directly addresses the topic."}}
            """
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            json_str = extract_json_from_response(response['message']['content'])
            if not json_str: raise ValueError("No JSON object found in snippet validator response.")
            data = json.loads(json_str)
            return data.get("is_relevant", False), data.get("reason", "No reason provided.")
        except Exception as e:
            self.progress_signal.emit((f"   - Snippet validation Error: {str(e)}", "ERROR"))
            return False, str(e)

# --- Worker Threads ---

class TopicExpansionWorker(QThread):
    progress = Signal(tuple)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, main_topic, target_topic_count, similarity_threshold, batch_size, generation_model_name, validation_model_name, client):
        super().__init__()
        self.main_topic, self.target_topic_count = main_topic, target_topic_count
        self.similarity_threshold, self.batch_size = similarity_threshold, batch_size
        self.generation_model_name = generation_model_name
        self.client = client
        self.relevance_validator = TopicRelevanceValidator(validation_model_name, self.progress, self.client)

    def run(self):
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                accepted_topics, accepted_embeddings, attempts = [], [], 0
                max_attempts = int(self.target_topic_count / self.batch_size * 5) + 20

                self.progress.emit((f"Starting topic generation with batch size {self.batch_size}. Target: {self.target_topic_count} topics.", "STAGE"))

                while len(accepted_topics) < self.target_topic_count and attempts < max_attempts:
                    attempts += 1
                    needed = self.target_topic_count - len(accepted_topics)
                    current_batch_size = min(self.batch_size, needed)
                    self.progress.emit((f"Starting batch {attempts}. Target: {self.target_topic_count}, Current: {len(accepted_topics)}, Requesting: {current_batch_size}", "INFO"))
                    
                    avoid_list_str = ", ".join(f"'{t}'" for t in accepted_topics[-100:])
                    
                    prompt = f"""
                    You are a Curriculum Design AI specializing in creating diverse and conceptually unique educational modules.

                    **Core Task:** Generate a list of sub-topics for the main topic provided.

                    **Main Topic:** "{self.main_topic}"

                    **Generation Guidelines & Constraints:**
                    1.  **Be Diverse:** Approach the main topic from multiple angles: historical, technological, social, ethical, and future-looking.
                    2.  **Be Concise:** EACH TOPIC MUST BE A CONCISE PHRASE, MAXIMUM 7-9 WORDS.
                    3.  **Quantity:** Generate exactly {current_batch_size} new, distinct sub-topics.

                    **Conceptual Areas to Avoid:**
                    The following topics represent themes that are already sufficiently covered. Do not generate topics that are too similar to these. DO NOT simply rephrase or create minor variations of topics in this list.
                    [{avoid_list_str}]

                    **Output Format:**
                    Your response MUST be a single, raw JSON object and nothing else. DO NOT add explanations, conversational text, or markdown formatting around the JSON object.
                    Example format: {{"topics": ["concise topic 1", "concise topic 2", ...]}}
                    """
                    
                    response = self.client.chat(
                        model=self.generation_model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'temperature': 0.9}
                    )
                    
                    json_str = extract_json_from_response(response['message']['content'])
                    if not json_str: self.progress.emit(("   - LLM response for batch contained no JSON. Retrying.", "WARN")); time.sleep(2); continue
                    
                    data = json.loads(json_str)
                    candidate_topics = data.get("topics", [])
                    if not candidate_topics: self.progress.emit(("   - LLM returned an empty topic list for the batch. Retrying.", "WARN")); continue

                    newly_added_count = 0
                    for candidate in candidate_topics:
                        if len(accepted_topics) >= self.target_topic_count: break
                        if not self.relevance_validator.is_relevant(self.main_topic, candidate): self.progress.emit((f"   - REJECTED (Irrelevant): '{candidate}'", "WARN")); continue
                            
                        candidate_embedding_response = self.client.embeddings(model='nomic-embed-text', prompt=candidate)
                        candidate_embedding = np.array(candidate_embedding_response['embedding'])
                        
                        is_too_similar = any(cosine_similarity(candidate_embedding, emb) > self.similarity_threshold for emb in accepted_embeddings)
                        if is_too_similar: self.progress.emit((f"   - REJECTED (Similarity): '{candidate}'", "WARN")); continue

                        self.progress.emit((f"   - ACCEPTED ({len(accepted_topics)+1}/{self.target_topic_count}): '{candidate}'", "SUCCESS"))
                        accepted_topics.append(candidate); accepted_embeddings.append(candidate_embedding)
                        newly_added_count += 1
                    
                    self.progress.emit((f"Batch complete. Added {newly_added_count} new unique topics.", "INFO"))
                
                if len(accepted_topics) < self.target_topic_count:
                    self.error.emit(f"Topic generation timed out. Produced {len(accepted_topics)} topics.")
                self.finished.emit(accepted_topics)
                return 
            
            except json.JSONDecodeError:
                self.progress.emit(("   - Failed to parse extracted JSON. The model may have returned a malformed response. Retrying batch.", "WARN"))
                time.sleep(2)
            except Exception as e:
                error_str = str(e).lower()
                if "timed out" in error_str and attempt < max_retries:
                    self.progress.emit((f"Network timeout detected. Retrying ({attempt + 1}/{max_retries})...", "WARN"))
                    time.sleep(3) 
                    continue
                else:
                    self.error.emit(f"A critical error occurred during topic expansion: {str(e)}")
                    return

class DataGenerationWorker(QThread):
    progress = Signal(tuple)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, sub_topics, total_samples, generation_model_name, validation_model_name, client):
        super().__init__()
        self.sub_topics, self.total_samples = sub_topics, total_samples
        self.generation_model_name = generation_model_name
        self.client = client
        self.validator = TextSnippetValidator(validation_model_name, self.progress, self.client)

    def run(self):
        dataset = []
        num_sub_topics = len(self.sub_topics)
        if num_sub_topics == 0: self.error.emit("Cannot generate data with an empty list of sub-topics."); return
        
        attempts, max_attempts = 0, self.total_samples * 3

        while len(dataset) < self.total_samples and attempts < max_attempts:
            sample_index, current_topic = len(dataset), self.sub_topics[len(dataset) % num_sub_topics]
            self.progress.emit((f"Attempting sample {sample_index + 1}/{self.total_samples} for topic: '{current_topic}'", "INFO"))
            
            try:
                prompt = f"""
                You are a Data Generation AI. Write a single, high-quality text snippet for the given sub-topic.
                Sub-Topic: "{current_topic}"
                Your response MUST be a single, raw JSON object and nothing else.
                Format: {{"text": "Your generated text here."}}
                """
                response = self.client.chat(model=self.generation_model_name, messages=[{'role': 'user', 'content': prompt}])
                json_str = extract_json_from_response(response['message']['content'])
                if not json_str: self.progress.emit(("   - Generation failed: No JSON found. Retrying.", "WARN")); attempts += 1; continue
                
                text_entry = json.loads(json_str)
                generated_text = text_entry.get("text", "")
                if not generated_text or len(generated_text) < 20: self.progress.emit(("   - Generation failed: Text too short. Retrying.", "WARN")); attempts += 1; continue

                is_relevant, reason = self.validator.validate_relevance(current_topic, generated_text)
                if not is_relevant: self.progress.emit((f"   - Snippet relevance FAILED. Reason: {reason}. Discarding.", "WARN")); attempts += 1; continue

                self.progress.emit((f"   - Generation and validation SUCCESSFUL for sample {sample_index + 1}.", "SUCCESS"))
                dataset.append({"input": current_topic, "output": generated_text})
            except Exception as e:
                self.progress.emit((f"   - Failed to generate or parse snippet: {e}. Retrying.", "WARN")); attempts += 1; time.sleep(2)
        
        if attempts >= max_attempts: self.error.emit(f"Failed to generate samples after {max_attempts} attempts.")
        self.finished.emit(dataset)

# --- Main Application UI ---
class SyntheticDatasetGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genisis-Mini")
        self.setWindowIcon(create_app_icon())
        self.setGeometry(100, 100, 800, 700)
        self.client = ollama.Client(timeout=45)
        self.critical_errors = []
        
        self.setStyleSheet("""
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
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12); main_layout.setSpacing(10)
        controls_group = QGroupBox("Controls"); controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(12, 20, 12, 12); controls_layout.setSpacing(12)
        form_layout = QFormLayout(); form_layout.setHorizontalSpacing(15); form_layout.setVerticalSpacing(10)
        
        self.main_topic_edit = QLineEdit("The history and impact of the Internet")
        self.target_topics_spin = QSpinBox(); self.target_topics_spin.setRange(10, 5000); self.target_topics_spin.setValue(50)
        self.total_samples_spin = QSpinBox(); self.total_samples_spin.setRange(1, 10000); self.total_samples_spin.setValue(100)
        self.similarity_threshold_spin = QDoubleSpinBox(); self.similarity_threshold_spin.setRange(0.7, 1.0); self.similarity_threshold_spin.setValue(0.95); self.similarity_threshold_spin.setSingleStep(0.01); self.similarity_threshold_spin.setDecimals(2)
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 50); self.batch_size_spin.setValue(7)
        self.generation_model_combo = QComboBox(); self.generation_model_combo.addItems(["granite4:tiny-h", "granite4:micro-h"])
        self.validation_model_combo = QComboBox(); self.validation_model_combo.addItems(["granite4:micro-h", "granite4:tiny-h"])
        self.export_format_combo = QComboBox(); self.export_format_combo.addItems(["JSONL (.jsonl)", "JSON Array (.json)"])

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
        log_header_layout.setContentsMargins(0,0,0,0)
        log_title_label = QLabel("Process Log"); log_title_label.setStyleSheet("font-weight: bold; color: #888; font-size: 12px; padding: 0 8px;")
        
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
        error_display_layout.setContentsMargins(2,8,2,2)
        self.error_text_edit = QTextEdit(); self.error_text_edit.setReadOnly(True); self.error_text_edit.setMaximumHeight(120)
        self.copy_error_btn = QPushButton("Copy Error Text"); self.copy_error_btn.setStyleSheet("padding: 6px; font-size: 11px;")
        self.copy_error_btn.clicked.connect(self.copy_error_text)
        error_display_layout.addWidget(self.error_text_edit)
        error_display_layout.addWidget(self.copy_error_btn)
        self.error_display_widget.hide()
        log_layout.addWidget(self.error_display_widget)

        self.output_edit = QTextEdit(); self.output_edit.setReadOnly(True)
        log_layout.addWidget(self.output_edit)
        main_layout.addWidget(self.log_group, 1)
        
    def log_message(self, message_tuple):
        message, level = message_tuple
        color_map = { "INFO": "#cccccc", "SUCCESS": "#2ECC71", "ERROR": "#E74C3C", "WARN": "#F39C12", "STAGE": "#3498DB", "VALIDATION": "#95a5a6" }
        color = color_map.get(level, "#cccccc")
        html = f"<hr><h4 style='color: {color};'>&raquo; {message}</h4>" if level == "STAGE" else f"<p style='color: {color}; margin: 2px 0;'>{message}</p>"
        self.output_edit.append(html)
        self.output_edit.verticalScrollBar().setValue(self.output_edit.verticalScrollBar().maximum())
    
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
        
        self.total_samples = self.total_samples_spin.value()
        self.generation_model = self.generation_model_combo.currentText()
        self.validation_model = self.validation_model_combo.currentText()
        
        self.topic_worker = TopicExpansionWorker(
            main_topic, self.target_topics_spin.value(), self.similarity_threshold_spin.value(),
            self.batch_size_spin.value(), self.generation_model, self.validation_model, self.client
        )
        self.topic_worker.progress.connect(self.log_message)
        self.topic_worker.finished.connect(self.on_topics_expanded)
        self.topic_worker.error.connect(self.generation_error)
        self.topic_worker.start()

    def on_topics_expanded(self, sub_topics):
        if not sub_topics: self.generation_error("Topic expansion resulted in an empty list. Cannot proceed."); return
        self.log_message((f"Topic Expansion Complete. Final unique topic count: {len(sub_topics)}", "SUCCESS"))
        self.log_message(("STAGE 2: DATASET GENERATION & VALIDATION", "STAGE"))
        
        self.data_worker = DataGenerationWorker(
            sub_topics, self.total_samples, self.generation_model, self.validation_model, self.client
        )
        self.data_worker.progress.connect(self.log_message)
        self.data_worker.finished.connect(self.generation_finished)
        self.data_worker.error.connect(self.generation_error)
        self.data_worker.start()
    
    def generation_finished(self, dataset):
        self.log_message(("PROCESS COMPLETE", "STAGE"))
        if not dataset:
            self.log_message(("Final dataset is empty. This may be due to repeated generation/validation failures.", "WARN"))
        else:
            export_format = self.export_format_combo.currentText()
            extension = ".jsonl" if "JSONL" in export_format else ".json"
            filename = get_next_filename(extension=extension)
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    if extension == ".jsonl":
                        for item in dataset: f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    else: json.dump(dataset, f, ensure_ascii=False, indent=4)
                self.log_message((f"Dataset successfully saved to {filename} ({len(dataset)} valid samples).", "SUCCESS"))
            except Exception as e:
                self.generation_error(f"Error saving file: {str(e)}")
        
        self.generate_btn.setEnabled(True)
    
    def generation_error(self, error_msg):
        self.log_message((f"FATAL ERROR: {error_msg}", "ERROR"))
        self.critical_errors.append(error_msg)
        
        error_count = len(self.critical_errors)
        plural = "s" if error_count > 1 else ""
        self.error_flag_btn.setText(f"View {error_count} Critical Error{plural}!")
        self.error_flag_btn.show()

        formatted_errors = "\n\n".join(f"- {e}" for e in self.critical_errors)
        self.error_text_edit.setText(formatted_errors)
        
        self.generate_btn.setEnabled(True)

    def toggle_error_display(self):
        self.error_display_widget.setVisible(not self.error_display_widget.isVisible())
    
    def copy_error_text(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.error_text_edit.toPlainText())
        self.log_message(("Error text copied to clipboard.", "INFO"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SyntheticDatasetGenerator()
    window.show()
    sys.exit(app.exec())