import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class StorageService:
    def __init__(self, recordings_dir: Path, outputs_dir: Path):
        self.recordings_dir = recordings_dir
        self.outputs_dir = outputs_dir

        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def build_session_paths(self, ts: str) -> Dict[str, Path]:
        return {
            "audio": self.recordings_dir / f"recording_{ts}.wav",
            "transcription": self.outputs_dir / f"transcription_{ts}.txt",
            "summary": self.outputs_dir / f"summary_{ts}.txt",
            "structured": self.outputs_dir / f"note_{ts}.json",
        }

    def save_text(self, path: Path, text: str):
        path.write_text(text, encoding="utf-8")

    def save_json(self, path: Path, data: Dict[str, Any]):
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")