import logging
from logging import Logger
from pathlib import Path

_LOGGERS = {}

def get_logger(name: str) -> Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("%(asctime)s - [%(levelname)s]: %(message)s")
    ch.setFormatter(ch_fmt)

    # File
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s: %(message)s")
    fh.setFormatter(fh_fmt)

    # Avoid duplicate handlers on reruns
    for h in list(logger.handlers):
        logger.removeHandler(h)

    logger.addHandler(ch)
    logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger
