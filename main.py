import sys
from PySide6.QtWidgets import QApplication
from app.ui import VoiceRecorderApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRecorderApp()
    window.show()
    sys.exit(app.exec())