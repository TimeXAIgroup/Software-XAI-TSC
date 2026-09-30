import logging
import sys
from pathlib import Path

logger = logging.getLogger()


def setup_logging(logger, out_path: Path, level: int = logging.INFO):
    # Configure formatter
    if level == logging.DEBUG:
        format = "%(asctime)s, %(name)s, %(levelname)s: %(message)s"
    else:
        format = "%(asctime)s, %(levelname)s: %(message)s"
    formatter = logging.Formatter(format)

    # Add stream handler to root logger
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level)

    # Create directory if it does not exist
    out_path.parent.mkdir(exist_ok=True, parents=True)
    logfile = out_path.with_suffix(".log")
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Finished setup, now also logging to file: %s", str(logfile))

    # Ignore messages:
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.pyplot").setLevel(logging.INFO)


def add_filehandler(logger, path):
    # Configure formatter
    format = "%(asctime)s, %(name)s, %(levelname)s: %(message)s"
    formatter = logging.Formatter(format)

    # Create directory if it does not exist
    path.parent.mkdir(exist_ok=True, parents=True)
    logfile = path.with_suffix(".log")
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Added a new FileHandler, now also logging to file: %s", str(logfile))
    return file_handler
