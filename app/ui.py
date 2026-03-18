import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QTextEdit,
    QTabWidget,
    QListWidget,
)

from app.config import (
    APP_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    SAMPLE_RATE,
    CHANNELS,
    RECORDINGS_DIR,
    OUTPUTS_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    OLLAMA_URL,
    OLLAMA_MODEL_NAME,
    DEFAULT_LANGUAGE,
)
from app.services.recorder import AudioRecorder
from app.services.transcriber import TranscriberService
from app.services.summarizer import SummarizerService
from app.services.storage import StorageService
from app.workers import TranscriptionWorker, SummaryWorker


class VoiceRecorderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.storage = StorageService(RECORDINGS_DIR, OUTPUTS_DIR)
        self.recorder = AudioRecorder(SAMPLE_RATE, CHANNELS)

        self.transcriber = None
        self.summarizer = SummarizerService(OLLAMA_URL, OLLAMA_MODEL_NAME)

        self.current_paths = {}
        self.last_transcription_text = ""
        self.last_structured_data = {}

        self.transcription_thread = None
        self.summary_thread = None

        self.status_label = None
        self.file_label = None
        self.start_button = None
        self.stop_button = None
        self.summary_button = None

        self.text_box = None
        self.summary_box = None
        self.json_box = None
        self.tasks_list = None
        self.points_list = None
        self.blockers_list = None
        self.keywords_list = None

        self.setWindowTitle(APP_TITLE)
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.init_ui()

        self.status_message("Loading Whisper model...", "orange")
        try:
            self.transcriber = TranscriberService(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            self.status_message("Status: Ready", "blue")
        except Exception as e:
            self.status_message("Status: Whisper load error", "red")
            QMessageBox.critical(self, "Whisper Error", str(e))

    def init_ui(self):
        title_label = QLabel("Local Voice to Text + Structured Summary")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: blue;")

        self.file_label = QLabel("No file generated")
        self.file_label.setAlignment(Qt.AlignCenter)
        self.file_label.setWordWrap(True)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_recording)
        self.start_button.setStyleSheet(
            "background-color: #2e7d32; color: white; padding: 10px; font-size: 14px;"
        )

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(
            "background-color: #c62828; color: white; padding: 10px; font-size: 14px;"
        )

        self.summary_button = QPushButton("Generate Structured Summary")
        self.summary_button.clicked.connect(self.generate_summary)
        self.summary_button.setEnabled(False)
        self.summary_button.setStyleSheet(
            "background-color: #1565c0; color: white; padding: 10px; font-size: 14px;"
        )

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setPlaceholderText("The transcription will appear here...")

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("The summary will appear here...")

        self.json_box = QTextEdit()
        self.json_box.setReadOnly(True)
        self.json_box.setPlaceholderText("The structured JSON will appear here...")

        self.tasks_list = QListWidget()
        self.points_list = QListWidget()
        self.blockers_list = QListWidget()
        self.keywords_list = QListWidget()

        transcription_tab = QWidget()
        transcription_layout = QVBoxLayout()
        transcription_layout.addWidget(QLabel("Transcription"))
        transcription_layout.addWidget(self.text_box)
        transcription_tab.setLayout(transcription_layout)

        summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        summary_layout.addWidget(QLabel("Summary"))
        summary_layout.addWidget(self.summary_box)
        summary_tab.setLayout(summary_layout)

        tasks_tab = QWidget()
        tasks_layout = QVBoxLayout()
        tasks_layout.addWidget(QLabel("Action Items"))
        tasks_layout.addWidget(self.tasks_list)
        tasks_tab.setLayout(tasks_layout)

        points_tab = QWidget()
        points_layout = QVBoxLayout()
        points_layout.addWidget(QLabel("Important Points"))
        points_layout.addWidget(self.points_list)
        points_tab.setLayout(points_layout)

        blockers_tab = QWidget()
        blockers_layout = QVBoxLayout()
        blockers_layout.addWidget(QLabel("Blockers"))
        blockers_layout.addWidget(self.blockers_list)
        blockers_tab.setLayout(blockers_layout)

        keywords_tab = QWidget()
        keywords_layout = QVBoxLayout()
        keywords_layout.addWidget(QLabel("Keywords"))
        keywords_layout.addWidget(self.keywords_list)
        keywords_tab.setLayout(keywords_layout)

        json_tab = QWidget()
        json_layout = QVBoxLayout()
        json_layout.addWidget(QLabel("Structured JSON"))
        json_layout.addWidget(self.json_box)
        json_tab.setLayout(json_layout)

        tabs = QTabWidget()
        tabs.addTab(transcription_tab, "Transcription")
        tabs.addTab(summary_tab, "Summary")
        tabs.addTab(tasks_tab, "Tasks")
        tabs.addTab(points_tab, "Important")
        tabs.addTab(blockers_tab, "Blockers")
        tabs.addTab(keywords_tab, "Keywords")
        tabs.addTab(json_tab, "JSON")

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.summary_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.status_label)
        main_layout.addSpacing(10)
        main_layout.addLayout(buttons_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.file_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(tabs)

        self.setLayout(main_layout)

    def status_message(self, text: str, color: str):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 14px; color: {color};")

    def clear_outputs(self):
        self.text_box.clear()
        self.summary_box.clear()
        self.json_box.clear()
        self.tasks_list.clear()
        self.points_list.clear()
        self.blockers_list.clear()
        self.keywords_list.clear()
        self.last_transcription_text = ""
        self.last_structured_data = {}

    def start_recording(self):
        try:
            self.clear_outputs()

            ts = self.storage.timestamp()
            self.current_paths = self.storage.build_session_paths(ts)

            self.recorder.start()

            self.status_message("Status: Recording...", "red")
            self.file_label.setText("Audio recording in progress...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.summary_button.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to start recording:\n{e}")

    def stop_recording(self):
        try:
            audio_data = self.recorder.stop()

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            if audio_data.size == 0:
                self.status_message("Status: Ready", "blue")
                self.file_label.setText("No audio captured")
                return

            self.recorder.save_wav(audio_data, str(self.current_paths["audio"]))
            self.file_label.setText(f"Audio saved:\n{self.current_paths['audio']}")
            self.status_message("Status: Transcribing...", "orange")

            self.run_transcription_worker(str(self.current_paths["audio"]))

        except Exception as e:
            self.status_message("Status: Error", "red")
            QMessageBox.critical(self, "Error", f"Error during stop/transcription:\n{e}")

    def run_transcription_worker(self, audio_path: str):
        self.transcription_thread = QThread()
        self.transcription_worker = TranscriptionWorker(
            self.transcriber,
            audio_path,
            language=DEFAULT_LANGUAGE,
        )
        self.transcription_worker.moveToThread(self.transcription_thread)

        self.transcription_thread.started.connect(self.transcription_worker.run)
        self.transcription_worker.finished.connect(self.on_transcription_finished)
        self.transcription_worker.error.connect(self.on_transcription_error)

        self.transcription_worker.finished.connect(self.transcription_thread.quit)
        self.transcription_worker.error.connect(self.transcription_thread.quit)
        self.transcription_worker.finished.connect(self.transcription_worker.deleteLater)
        self.transcription_worker.error.connect(self.transcription_worker.deleteLater)
        self.transcription_thread.finished.connect(self.transcription_thread.deleteLater)

        self.transcription_thread.start()

    def on_transcription_finished(self, text: str):
        self.last_transcription_text = text.strip()

        if self.last_transcription_text:
            self.text_box.setPlainText(self.last_transcription_text)
            self.storage.save_text(self.current_paths["transcription"], self.last_transcription_text)
            self.summary_button.setEnabled(True)

            self.status_message("Status: Transcription completed", "green")
            self.file_label.setText(
                f"Audio: {self.current_paths['audio']}\n"
                f"Transcription: {self.current_paths['transcription']}"
            )
        else:
            self.text_box.setPlainText("[No text detected]")
            self.summary_button.setEnabled(False)
            self.status_message("Status: No speech detected", "orange")

    def on_transcription_error(self, error_message: str):
        self.status_message("Status: Transcription error", "red")
        self.summary_button.setEnabled(False)
        QMessageBox.critical(self, "Transcription Error", error_message)

    def generate_summary(self):
        transcription = self.last_transcription_text.strip()
        if not transcription:
            QMessageBox.warning(self, "Warning", "No transcription available.")
            return

        self.status_message("Status: Generating structured summary...", "orange")
        self.summary_button.setEnabled(False)

        self.run_summary_worker(transcription)

    def run_summary_worker(self, transcription: str):
        self.summary_thread = QThread()
        self.summary_worker = SummaryWorker(self.summarizer, transcription)
        self.summary_worker.moveToThread(self.summary_thread)

        self.summary_thread.started.connect(self.summary_worker.run)
        self.summary_worker.finished.connect(self.on_summary_finished)
        self.summary_worker.error.connect(self.on_summary_error)

        self.summary_worker.finished.connect(self.summary_thread.quit)
        self.summary_worker.error.connect(self.summary_thread.quit)
        self.summary_worker.finished.connect(self.summary_worker.deleteLater)
        self.summary_worker.error.connect(self.summary_worker.deleteLater)
        self.summary_thread.finished.connect(self.summary_thread.deleteLater)

        self.summary_thread.start()

    def _fill_list_widget(self, widget, items):
        widget.clear()
        for item in items:
            widget.addItem(str(item))

    def on_summary_finished(self, structured_data: dict):
        self.last_structured_data = structured_data

        summary_text = structured_data.get("summary", "")
        action_items = structured_data.get("action_items", [])
        important_points = structured_data.get("important_points", [])
        blockers = structured_data.get("blockers", [])
        keywords = structured_data.get("keywords", [])
        priority = structured_data.get("priority", "medium")

        self.summary_box.setPlainText(
            f"{summary_text}\n\nPriority: {priority}"
        )
        self.json_box.setPlainText(
            json.dumps(structured_data, indent=2, ensure_ascii=False)
        )

        self._fill_list_widget(self.tasks_list, action_items)
        self._fill_list_widget(self.points_list, important_points)
        self._fill_list_widget(self.blockers_list, blockers)
        self._fill_list_widget(self.keywords_list, keywords)

        self.storage.save_text(self.current_paths["summary"], summary_text)
        self.storage.save_json(self.current_paths["structured"], structured_data)

        self.status_message("Status: Structured summary completed", "green")
        self.file_label.setText(
            f"Audio: {self.current_paths['audio']}\n"
            f"Transcription: {self.current_paths['transcription']}\n"
            f"Summary: {self.current_paths['summary']}\n"
            f"JSON: {self.current_paths['structured']}"
        )

        self.summary_button.setEnabled(True)

    def on_summary_error(self, error_message: str):
        self.summary_button.setEnabled(True)
        self.status_message("Status: LLM error", "red")
        QMessageBox.critical(
            self,
            "LLM Error",
            f"{error_message}\n\nCheck that Ollama is installed and running."
        )