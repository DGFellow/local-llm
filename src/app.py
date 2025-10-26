import sys, os
from PyQt6.QtWidgets import QApplication
from dotenv import load_dotenv

from .config import Config
from .utils.logging import setup_logging
from .ui.main_window import MainWindow

def main():
    setup_logging()
    load_dotenv()  # optional .env overrides

    app = QApplication(sys.argv)
    cfg = Config()
    win = MainWindow(cfg)
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()