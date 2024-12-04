# logging_setup.py

import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(session_id: str):
    """Setup main logging with RotatingFileHandler."""
    logger = logging.getLogger("OpenAIClient")
    logger.setLevel(logging.DEBUG)

    # Ensure log directories exist
    os.makedirs('output/logs', exist_ok=True)
    os.makedirs('output/logs/muxing', exist_ok=True)

    # File handler
    file_handler = RotatingFileHandler(f"output/logs/app_{session_id}.log", maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Muxing Logger
    muxing_logger = logging.getLogger("OpenAIClient_muxing")
    muxing_logger.setLevel(logging.DEBUG)
    muxing_file_handler = RotatingFileHandler(f"output/logs/muxing/muxing_{session_id}.log", maxBytes=10*1024*1024, backupCount=5)
    muxing_file_handler.setFormatter(formatter)
    muxing_logger.addHandler(muxing_file_handler)

    logger.info("Logging initialized.")
    muxing_logger.info("Muxing logging initialized.")

    return logger, muxing_logger
