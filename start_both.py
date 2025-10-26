"""
Launch both Flask API server and PyQt6 GUI simultaneously
"""
import sys
import threading
import logging
from PyQt6.QtWidgets import QApplication
from dotenv import load_dotenv

# Import from your existing code
from src.config import Config
from src.utils.logging import setup_logging
from src.ui.main_window import MainWindow

# Import API server
import api_server

def run_api_server():
    """Run Flask server in background thread"""
    api_server.main()

def main():
    """Launch both API and GUI"""
    setup_logging()
    load_dotenv()
    
    print("\n" + "="*60)
    print("🚀 Starting Local LLM (API + GUI)")
    print("="*60 + "\n")
    
    # Start API server in daemon thread
    print("📡 Starting Flask API server...")
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Give API time to initialize
    import time
    time.sleep(3)
    
    # Start GUI in main thread
    print("🖥️  Starting GUI...")
    app = QApplication(sys.argv)
    cfg = Config()
    win = MainWindow(cfg)
    win.resize(900, 700)
    win.setWindowTitle("Local LLM - API Server Running on :5000")
    win.show()
    
    print("\n✅ Both API and GUI are running!")
    print("   API: http://localhost:5000")
    print("   GUI: Active window\n")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()