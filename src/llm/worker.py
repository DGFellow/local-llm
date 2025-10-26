from PyQt6.QtCore import QObject, QThread, pyqtSignal
from typing import List, Tuple
from .engine import LLMEngine

class GenerateWorker(QObject):
    token_signal = pyqtSignal(str)     # emits incremental text
    done_signal = pyqtSignal()         # signals completion
    error_signal = pyqtSignal(str)     # emits error string

    def __init__(self, engine: LLMEngine, system_prompt: str, history: List[Tuple[str, str]], user_msg: str):
        super().__init__()
        self.engine = engine
        self.system_prompt = system_prompt
        self.history = history
        self.user_msg = user_msg

    def run(self):
        try:
            for chunk in self.engine.generate_stream(self.system_prompt, self.history, self.user_msg):
                self.token_signal.emit(chunk)
            self.done_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))