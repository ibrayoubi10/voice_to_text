import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


class AudioRecorder:
    def __init__(self, sample_rate: int, channels: int):
        self.sample_rate = sample_rate
        self.channels = channels

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recorded_frames = []
        self.stream: Optional[sd.InputStream] = None
        self.collect_thread: Optional[threading.Thread] = None

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def _collect_audio(self):
        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.recorded_frames.append(data)
            except queue.Empty:
                continue

    def start(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_queue = queue.Queue()
        self.recorded_frames = []

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._audio_callback,
        )
        self.stream.start()

        self.collect_thread = threading.Thread(target=self._collect_audio, daemon=True)
        self.collect_thread.start()

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            return np.array([], dtype=np.int16)

        self.is_recording = False

        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.recorded_frames:
            return np.array([], dtype=np.int16)

        return np.concatenate(self.recorded_frames, axis=0)

    def save_wav(self, audio_data: np.ndarray, output_path: str):
        write(output_path, self.sample_rate, audio_data)