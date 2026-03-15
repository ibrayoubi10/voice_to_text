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


# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
RECORDINGS_DIR = "recordings"
OUTPUTS_DIR = "outputs"
MODEL_SIZE = "base"   # tiny, base, small, medium


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
        self.collect_thread = None

        self.setWindowTitle("Vocal to Texte - Local")
        self.setFixedSize(700, 500)

        self.status_label = None
        self.file_label = None
        self.start_button = None
        self.stop_button = None
        self.text_box = None

        self.init_ui()

        # Charger le modèle une seule fois
        self.status_message("Charging Whisper model...", "orange")
        self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        self.status_message("Statut : prêt", "blue")

    def init_ui(self):
        title_label = QLabel("Local Voice to Text")
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

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setPlaceholderText("The transcription will appear here...")

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.status_label)
        main_layout.addSpacing(10)
        main_layout.addLayout(buttons_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.file_label)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.text_box)

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

            self.status_message("Statut : enregistrement...", "red")
            self.file_label.setText("Enregistrement audio en cours...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

            self.collect_thread = threading.Thread(target=self.collect_audio, daemon=True)
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
                return

            # 1) Sauvegarder l'audio
            audio_data = np.concatenate(self.recorded_frames, axis=0)
            write(self.current_audio_path, SAMPLE_RATE, audio_data)

            self.status_message("Statut : transcription...", "orange")
            self.file_label.setText(f"Audio sauvegardé :\n{self.current_audio_path}")

            # 2) Transcrire
            text = self.transcribe_audio(self.current_audio_path)

            # 3) Afficher le texte
            if text:
                self.text_box.setPlainText(text)
            else:
                self.text_box.setPlainText("[Aucun texte détecté]")

            # 4) Sauvegarder le texte
            txt_path = self.save_transcription(text if text else "")

            self.status_message("Statut : terminé", "green")
            self.file_label.setText(
                f"Audio : {self.current_audio_path}\nTexte : {txt_path}"
            )

            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

            QMessageBox.information(
                self,
                "Succès",
                f"Transcription terminée.\n\nFichier texte :\n{txt_path}"
            )

        except Exception as e:
            self.status_message("Statut : erreur", "orange")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            QMessageBox.critical(self, "Erreur", f"Erreur pendant la transcription :\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRecorderApp()
    window.show()
    sys.exit(app.exec())