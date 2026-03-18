from faster_whisper import WhisperModel


class TranscriberService:
    def __init__(self, model_size: str, device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str = "fr") -> str:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
        )

        text_parts = []
        for segment in segments:
            cleaned = segment.text.strip()
            if cleaned:
                text_parts.append(cleaned)

        return " ".join(text_parts).strip()