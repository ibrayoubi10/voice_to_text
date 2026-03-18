from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import QObject, Signal, Slot


class TranscriptionWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, transcriber_service, audio_path: str, language: str = "fr"):
        super().__init__()
        self.transcriber_service = transcriber_service
        self.audio_path = audio_path
        self.language = language

    @Slot()
    def run(self):
        try:
            text = self.transcriber_service.transcribe(self.audio_path, language=self.language)
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class SummaryWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, summarizer_service, transcription: str):
        super().__init__()
        self.summarizer_service = summarizer_service
        self.transcription = transcription

    @Slot()
    def run(self):
        try:
            structured = self.summarizer_service.summarize_structured(self.transcription)
            self.finished.emit(structured)
        except Exception as e:
            self.error.emit(str(e))