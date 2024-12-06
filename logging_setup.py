# logging_setup.py

import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(session_id: str):
    """
    Setup main logging with RotatingFileHandler for the OpenAIClient and Muxer.

    Args:
        session_id (str): Unique identifier for the session to segregate logs.
    
    Returns:
        tuple: A tuple containing the main logger and the muxing logger.
    """
    # Define logger names with session_id for better log segregation
    main_logger_name = f"OpenAIClient-{session_id}"
    muxing_logger_name = f"Muxer-{session_id}"
    rtmp_logger_name = f"RTMPStreamer-{session_id}"  # Optional: Separate logger for RTMPStreamer

    # Create main logger
    main_logger = logging.getLogger(main_logger_name)
    main_logger.setLevel(logging.DEBUG)  # Capture all levels DEBUG and above

    # Create muxing logger
    muxing_logger = logging.getLogger(muxing_logger_name)
    muxing_logger.setLevel(logging.DEBUG)  # Capture all levels DEBUG and above

    # Optionally, create RTMPStreamer logger
    # rtmp_logger = logging.getLogger(rtmp_logger_name)
    # rtmp_logger.setLevel(logging.DEBUG)

    # Ensure log directories exist
    log_dir = 'output/logs'
    muxing_log_dir = os.path.join(log_dir, 'muxing')
    rtmp_log_dir = os.path.join(log_dir, 'rtmp')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(muxing_log_dir, exist_ok=True)
    os.makedirs(rtmp_log_dir, exist_ok=True)

    # Formatter for file handlers
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Formatter for console handler
    console_formatter = logging.Formatter(
        fmt='%(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Setup RotatingFileHandler for main logger
    main_file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, f"client_{session_id}.log"),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(file_formatter)
    main_logger.addHandler(main_file_handler)

    # Setup RotatingFileHandler for muxing logger
    muxing_file_handler = RotatingFileHandler(
        filename=os.path.join(muxing_log_dir, f"muxing_{session_id}.log"),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    muxing_file_handler.setLevel(logging.DEBUG)
    muxing_file_handler.setFormatter(file_formatter)
    muxing_logger.addHandler(muxing_file_handler)

    rtmp_logger = logging.getLogger(rtmp_logger_name)
    rtmp_file_handler = RotatingFileHandler(
        filename=os.path.join(rtmp_log_dir, f"rtmp_{session_id}.log"),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    rtmp_file_handler.setLevel(logging.DEBUG)
    rtmp_file_handler.setFormatter(file_formatter)
    rtmp_logger.addHandler(rtmp_file_handler)

    # Setup Console Handler for main logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set higher level to reduce console verbosity
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)

    # Prevent loggers from propagating to the root logger to avoid duplicate logs
    main_logger.propagate = False
    muxing_logger.propagate = False
    rtmp_logger.propagate = False

    # Initial log messages to confirm setup
    main_logger.info("Logging initialized for OpenAIClient.")
    muxing_logger.info("Logging initialized for Muxer.")
    rtmp_logger.info("Logging initialized for RTMPStreamer.") 

    return main_logger, muxing_logger, rtmp_logger
