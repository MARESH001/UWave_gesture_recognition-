import logging
import os
from datetime import datetime

# Create a directory for logs
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Create a timestamped log file path
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# Configure the logger
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Log file path
    level=logging.INFO,  # Log level (could be dynamic based on use case)
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(logger_name):
    """Return a logger instance with the specified name."""
    logger = logging.getLogger(logger_name)

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

# Example of getting a logger and logging a message
logger = get_logger(__name__)
logger.info("Logging system initialized successfully.")
