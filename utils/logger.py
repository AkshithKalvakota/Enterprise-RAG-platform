import logging
import sys

def setup_logger(name="enterprise_rag"):
    """
    Sets up a standardized logger for the enterprise application.
    Outputs logs with timestamps and severity levels.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is called multiple times
    if not logger.handlers:
        # Console Handler (Prints to the terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Create a professional format: [TIME] - [NAME] - [LEVEL] - [MESSAGE]
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger