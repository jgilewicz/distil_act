import logging
import os
import sys


class Logger:
    def __init__(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename), exist_ok=True) if os.path.dirname(
            filename
        ) else None

        fmt = logging.Formatter(
            "[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S"
        )

        self._log = logging.getLogger(filename)
        self._log.setLevel(logging.DEBUG)

        fh = logging.FileHandler(filename)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)

        self._log.addHandler(fh)
        self._log.addHandler(sh)

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)
