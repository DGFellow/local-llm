from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout,
    QMessageBox, QLabel, QComboBox, QDialog, QDialogButtonBox, QMenuBar
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer, Qt
from PyQt6.QtGui import QAction
from src.config import Config, save_env
from src.llm.engine import LLMEngine
from src.llm.worker import GenerateWorker


class SettingsDialog(QDialog):
    def __init__(self, parent, cfg: Config):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self.model_combo = QComboBox(self)
        models = [
            "Phi-3.5-mini-instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
        self.model_combo.addItems(models)
        try:
            idx = models.index(cfg.model_id)
            self.model_combo.setCurrentIndex(idx)
        except ValueError:
            self.model_combo.insertItem(0, cfg.model_id)
            self.model_combo.setCurrentIndex(0)

        self.precision_combo = QComboBox(self)
        self.precision_combo.addItems(["auto", "fp16"])
        try:
            self.precision_combo.setCurrentIndex(["auto", "fp16"].index(cfg.precision))
        except ValueError:
            pass

        layout = QVBoxLayout(self)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model:"))
        row1.addWidget(self.model_combo)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Precision:"))
        row2.addWidget(self.precision_combo)

        layout.addLayout(row1)
        layout.addLayout(row2)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        return self.model_combo.currentText(), self.precision_combo.currentText()


class LoadWorker(QObject):
    success_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, engine: LLMEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        try:
            self.engine.load()
            self.success_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QWidget):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Menu Bar
        self.menu_bar = QMenuBar(self)
        file_menu = self.menu_bar.addMenu("Settings")
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.open_settings)
        file_menu.addAction(settings_action)

        # UI
        self.chat = QTextEdit(self)
        self.chat.setReadOnly(True)
        self.input = QTextEdit(self)
        self.input.setAcceptRichText(False)
        self.input.setFixedHeight(30)  # Approximate single-line height
        self.input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)  # Enable wrapping
        self.input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # Show scrollbar if needed
        self.send_btn = QPushButton("Send", self)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.input)
        btn_row.addWidget(self.send_btn)

        root = QVBoxLayout()
        root.addWidget(self.menu_bar)
        root.addWidget(self.chat)
        root.addLayout(btn_row)
        self.setLayout(root)

        # Window title reflects current cfg
        self.update_title()

        # LLM
        self.engine = LLMEngine(cfg)
        self.history: list[tuple[str, str]] = []  # [(user, assistant)]
        self._last_user_msg: str | None = None
        self._current_reply: str = ""
        self._reply_cursor = None     # QTextCursor for in-place updates
        self._reply_block_pos = None  # int: starting position of this reply block
        self._thinking_timer = QTimer(self)  # Timer for thinking animation
        self._thinking_dots = 0  # Counter for dots in animation

        # Wire up events
        self.send_btn.clicked.connect(self.on_send)
        self.input.textChanged.connect(self.check_input)  # Enable send on valid input
        self._thinking_timer.timeout.connect(self.update_thinking_animation)

        # Show initial messages and start background model load
        self.append_sys(f"Loading model: {self.cfg.model_id} ...")
        self.send_btn.setEnabled(False)
        self.start_loader_thread()

        # Final window tweaks
        self.update_title()

    # ===== Utility UI methods =====
    def update_title(self):
        self.setWindowTitle(f"HF Local Chat (PyQt6) — {self.cfg.model_id} [{self.cfg.precision}]")

    def append_sys(self, text: str):
        self.chat.append(f"<i>{text}</i>")
        self.chat.ensureCursorVisible()

    def append_user(self, text: str):
        self.chat.append(f"<b>You:</b> {text}")
        self.chat.ensureCursorVisible()

    def append_assistant_line(self, text: str):
        cursor = self.chat.textCursor()
        if self._reply_block_pos is None:
            # Start a new block
            self.chat.append("")
            self._reply_block_pos = cursor.position()
            self._reply_cursor = cursor
        else:
            # Move to the reply block
            cursor.setPosition(self._reply_block_pos)
            cursor.movePosition(cursor.MoveOperation.EndOfBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
        # Insert new text
        cursor.insertHtml(f"<b>Assistant:</b> {text}")
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()

    def update_thinking_animation(self):
        self._thinking_dots = (self._thinking_dots + 1) % 4
        dots = "." * self._thinking_dots
        self.append_assistant_line(f"Thinking{dots}")

    # ===== Threaded model loading =====
    def start_loader_thread(self):
        self.loader_thread = QThread(self)
        self.loader = LoadWorker(self.engine)
        self.loader.moveToThread(self.loader_thread)

        self.loader_thread.started.connect(self.loader.run)
        self.loader.success_signal.connect(self.on_model_ready)
        self.loader.error_signal.connect(self.on_error)

        # Cleanup
        self.loader.success_signal.connect(self.loader_thread.quit)
        self.loader.success_signal.connect(self.loader.deleteLater)
        self.loader_thread.finished.connect(self.loader_thread.deleteLater)

        QTimer.singleShot(0, lambda: self.loader_thread.start())

    def on_model_ready(self):
        self.append_sys("Model ready.")
        self.send_btn.setEnabled(True)

    # ===== Sending & streaming =====
    def check_input(self):
        # Enable send button only if input is not empty
        self.send_btn.setEnabled(bool(self.input.toPlainText().strip()))

    def on_send(self):
        msg = self.input.toPlainText().strip()
        if not msg:
            return
        self.send_btn.setEnabled(False)
        self.input.setReadOnly(True)  # Lock input during generation

        self.input.clear()
        self.append_user(msg)
        self._last_user_msg = msg

        # Start thinking animation
        self._reply_block_pos = None
        self._current_reply = ""
        self._thinking_timer.start(500)  # Update every 500ms

        # Start generation worker
        self.worker_thread = QThread(self)
        self.worker = GenerateWorker(self.engine, self.cfg.chat_system_prompt, self.history, msg)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.token_signal.connect(self.on_token)
        self.worker.done_signal.connect(self.on_done)
        self.worker.error_signal.connect(self.on_error)

        # Cleanup
        self.worker.done_signal.connect(self.worker_thread.quit)
        self.worker.done_signal.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def on_token(self, text: str):
        # Accumulate tokens internally, don't update UI yet
        self._current_reply += text

    def on_done(self):
        # Stop thinking animation and show final reply
        self._thinking_timer.stop()
        self.append_assistant_line(self._current_reply)
        if self._last_user_msg is not None:
            self.history.append((self._last_user_msg, self._current_reply))
        self.send_btn.setEnabled(True)
        self.input.setReadOnly(False)

    # ===== Settings & reload =====
    def open_settings(self):
        dlg = SettingsDialog(self, self.cfg)
        if dlg.exec():
            model_id, precision = dlg.values()
            save_env(model_id, precision)
            self.cfg.model_id = model_id
            self.cfg.precision = precision
            self.update_title()

            # Reload model in background
            self.append_sys(f"Reloading model: {self.cfg.model_id} ({self.cfg.precision}) ...")
            self.send_btn.setEnabled(False)

            self.engine = LLMEngine(self.cfg)
            self.start_loader_thread()

    # ===== Error handling =====
    def on_error(self, err: str):
        self._thinking_timer.stop()
        self.send_btn.setEnabled(True)
        self.input.setReadOnly(False)
        QMessageBox.critical(self, "Error", err)