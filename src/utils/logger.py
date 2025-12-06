import logging
import sys
import os

def setup_logger():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)

    logger.info(f"Logger initialised with level: {log_level}")

    return logger