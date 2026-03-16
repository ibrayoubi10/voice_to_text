import os
import sys
import queue
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QTextEdit,
)
from PySide6.QtCore import Qt

from summarizer import summarize_text


# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
RECORDINGS_DIR = "recordings"
OUTPUTS_DIR = "outputs"
MODEL_SIZE = "base"          # tiny, base, small, medium
LLM_MODEL_NAME = "llama3"    # modèle Ollama


class VoiceRecorderApp(QWidget):
    def __init__(self):
        super().__init__()

        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recorded_frames = []
        self.stream = None
        self.current_audio_path = None
        self.current_text_path = None
        self.current_summary_path = None
        self.collect_thread = None
        self.last_transcription_text = ""

        self.setWindowTitle("Voice to Text + Summary - Local")
        self.setFixedSize(800, 700)

        self.status_label = None
        self.file_label = None
        self.start_button = None
        self.stop_button = None
        self.summary_button = None
        self.text_box = None
        self.summary_box = None

        self.init_ui()

        # Charger Whisper une seule fois
        self.status_message("Charging Whisper model...", "orange")
        self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        self.status_message("Status: Ready", "blue")

    def init_ui(self):
        title_label = QLabel("Local Voice to Text + LLM Summary")
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

        self.summary_button = QPushButton("Generate Summary")
        self.summary_button.clicked.connect(self.generate_summary)
        self.summary_button.setEnabled(False)
        self.summary_button.setStyleSheet(
            "background-color: #1565c0; color: white; padding: 10px; font-size: 14px;"
        )

        transcription_label = QLabel("Transcription")
        transcription_label.setStyleSheet("font-size: 15px; font-weight: bold;")

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setPlaceholderText("The transcription will appear here...")

        summary_label = QLabel("Summary")
        summary_label.setStyleSheet("font-size: 15px; font-weight: bold;")

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("The summary will appear here...")

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
        main_layout.addWidget(transcription_label)
        main_layout.addWidget(self.text_box)
        main_layout.addSpacing(10)
        main_layout.addWidget(summary_label)
        main_layout.addWidget(self.summary_box)

        self.setLayout(main_layout)

    def status_message(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-size: 14px; color: {color};")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.recorded_frames = []
        self.audio_queue = queue.Queue()
        self.text_box.clear()
        self.summary_box.clear()
        self.last_transcription_text = ""
        self.summary_button.setEnabled(False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_audio_path = os.path.join(
            RECORDINGS_DIR, f"recording_{timestamp}.wav"
        )

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                callback=self.audio_callback
            )
            self.stream.start()

            self.status_message("Status: Recording...", "red")
            self.file_label.setText("Audio recording in progress...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.collect_thread = threading.Thread(
                target=self.collect_audio,
                daemon=True
            )
            self.collect_thread.start()

        except Exception as e:
            self.is_recording = False
            QMessageBox.critical(self, "Error", f"Unable to start recording:\n{e}")

    def collect_audio(self):
        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.recorded_frames.append(data)
            except queue.Empty:
                continue

    def transcribe_audio(self, audio_path):
        segments, info = self.model.transcribe(
            audio_path,
            language="fr",
            vad_filter=True
        )

        text_parts = []
        for segment in segments:
            cleaned = segment.text.strip()
            if cleaned:
                text_parts.append(cleaned)

        return " ".join(text_parts).strip()

    def save_transcription(self, text):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_text_path = os.path.join(
            OUTPUTS_DIR, f"transcription_{timestamp}.txt"
        )

        with open(self.current_text_path, "w", encoding="utf-8") as f:
            f.write(text)

        return self.current_text_path

    def save_summary(self, summary):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_summary_path = os.path.join(
            OUTPUTS_DIR, f"summary_{timestamp}.txt"
        )

        with open(self.current_summary_path, "w", encoding="utf-8") as f:
            f.write(summary)

        return self.current_summary_path

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            if not self.recorded_frames:
                self.status_message("Status: Ready", "blue")
                self.file_label.setText("No audio captured")
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.summary_button.setEnabled(False)
                return

            # 1) Sauvegarder l'audio
            audio_data = np.concatenate(self.recorded_frames, axis=0)
            write(self.current_audio_path, SAMPLE_RATE, audio_data)

            self.status_message("Status: Transcribing...", "orange")
            self.file_label.setText(f"Audio saved:\n{self.current_audio_path}")

            # 2) Transcrire
            text = self.transcribe_audio(self.current_audio_path)
            self.last_transcription_text = text if text else ""

            # 3) Afficher le texte
            if text:
                self.text_box.setPlainText(text)
                self.summary_button.setEnabled(True)
            else:
                self.text_box.setPlainText("[No text detected]")
                self.summary_button.setEnabled(False)

            # 4) Sauvegarder le texte
            txt_path = self.save_transcription(self.last_transcription_text)

            self.status_message("Status: Transcription completed", "green")
            self.file_label.setText(
                f"Audio: {self.current_audio_path}\nTranscription: {txt_path}"
            )

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            QMessageBox.information(
                self,
                "Success",
                f"Transcription completed.\n\nText file:\n{txt_path}"
            )

        except Exception as e:
            self.status_message("Status: Error", "orange")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.summary_button.setEnabled(False)
            QMessageBox.critical(self, "Error", f"Error during transcription:\n{e}")

    def generate_summary(self):
        transcription = self.last_transcription_text.strip()

        if not transcription:
            QMessageBox.warning(
                self,
                "Warning",
                "No transcription available. Please record and transcribe audio first."
            )
            return

        try:
            self.status_message("Status: Generating summary with LLM...", "orange")
            self.summary_button.setEnabled(False)

            summary = summarize_text(transcription, model=LLM_MODEL_NAME)

            if not summary.strip():
                summary = "[No summary generated]"

            self.summary_box.setPlainText(summary)
            summary_path = self.save_summary(summary)

            self.status_message("Status: Summary completed", "green")
            self.file_label.setText(
                f"Audio: {self.current_audio_path}\n"
                f"Transcription: {self.current_text_path}\n"
                f"Summary: {summary_path}"
            )

            QMessageBox.information(
                self,
                "Success",
                f"Summary generated successfully.\n\nSummary file:\n{summary_path}"
            )

        except Exception as e:
            self.status_message("Status: LLM error", "orange")
            QMessageBox.critical(
                self,
                "LLM Error",
                f"Error while generating summary:\n{e}\n\n"
                "Check that Ollama is installed and running."
            )
        finally:
            self.summary_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRecorderApp()
    window.show()
    sys.exit(app.exec())