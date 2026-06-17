from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure a shared logging format for CLI entrypoints."""
    fmt = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(file_handler)
