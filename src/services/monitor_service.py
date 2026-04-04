import time
import os
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='folder_monitoring.log')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class ProjectMonitorHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            # Avoid circular logging loops
            if event.src_path.endswith((".log", ".json", ".tmp", ".pyc")):
                return
            if "__pycache__" in event.src_path:
                return

            logging.info(f"File modified: {event.src_path}")
            # If a new model checkpoint is saved, log it
            if "models" in event.src_path and event.src_path.endswith(".h5"):
                logging.info(f"New model checkpoint detected: {os.path.basename(event.src_path)}")

    def on_created(self, event):
        if not event.is_directory:
            if event.src_path.endswith((".log", ".json", ".tmp", ".pyc")):
                return
            if "__pycache__" in event.src_path:
                return
                
            logging.info(f"File created: {event.src_path}")
            # If a new data file is added to Data folder
            if "Data" in event.src_path and event.src_path.endswith((".csv", ".json", ".xlsx")):
                logging.info(f"New data source found: {os.path.basename(event.src_path)}. Ready for analysis.")

    def on_deleted(self, event):
        if not event.is_directory:
            logging.info(f"File deleted: {event.src_path}")

def start_monitor(path_to_watch):
    event_handler = ProjectMonitorHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=True)
    observer.start()
    logging.info(f"Monitoring started for: {path_to_watch}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    start_monitor(PROJECT_ROOT)
