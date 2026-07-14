"""
utils.py — Shared utilities for logging, folder management, and reporting.
American Express Campus Challenge 2026 | Modelling Formulation Project
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict


def setup_experiment_folder(experiment_dir: str) -> str:
    """
    Creates the unique timestamped experiment folder and all subfolders.

    Why: Every experiment must be isolated. Nothing is overwritten.
    Each run produces a fresh folder with full audit trail.

    Args:
        experiment_dir: Absolute path to the experiment directory.

    Returns:
        experiment_dir (confirmed created).

    Raises:
        RuntimeError if folder cannot be created.
    """
    subfolders = ["data", "reports", "visualizations", "models", "submissions", "logs"]
    try:
        for sub in subfolders:
            os.makedirs(os.path.join(experiment_dir, sub), exist_ok=True)
        return experiment_dir
    except Exception as e:
        raise RuntimeError(f"[FAIL] Could not create experiment folder: {e}")


def setup_logger(experiment_dir: str, name: str = "pipeline") -> logging.Logger:
    """
    Sets up a logger that writes to both console and a log file.

    Why: Silent failures are forbidden. Every action must be logged
    with a timestamp for full reproducibility and debugging.

    Args:
        experiment_dir: Root experiment directory.
        name: Logger name identifier.

    Returns:
        Configured Logger instance.
    """
    log_path = os.path.join(experiment_dir, "logs", f"{name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_experiment_log(
    experiment_dir: str,
    log_data: Dict[str, Any],
    filename: str = "experiment_log.json"
) -> None:
    """
    Saves a structured JSON experiment log to the logs folder.

    Why: Every experiment must be reproducible and traceable.
    The log captures inputs, parameters, timestamps, and outcomes.

    Args:
        experiment_dir: Root experiment directory.
        log_data: Dictionary of experiment metadata.
        filename: Output filename.
    """
    log_path = os.path.join(experiment_dir, "logs", filename)
    log_data["saved_at"] = datetime.now().isoformat()
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)


def save_text_report(
    experiment_dir: str,
    content: str,
    subfolder: str,
    filename: str
) -> str:
    """
    Saves a text report to the specified subfolder.

    Args:
        experiment_dir: Root experiment directory.
        content: Report text content.
        subfolder: Subfolder inside experiment_dir.
        filename: Output filename.

    Returns:
        Absolute path to the saved file.
    """
    path = os.path.join(experiment_dir, subfolder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def checkpoint_pass(name: str, logger: logging.Logger) -> None:
    """Logs a checkpoint PASS."""
    logger.info(f"{'='*60}")
    logger.info(f"CHECKPOINT: {name} — STATUS: PASS ✓")
    logger.info(f"{'='*60}")


def checkpoint_fail(name: str, reason: str, logger: logging.Logger) -> None:
    """Logs a checkpoint FAIL and raises immediately."""
    logger.error(f"{'='*60}")
    logger.error(f"CHECKPOINT: {name} — STATUS: FAIL ✗")
    logger.error(f"REASON: {reason}")
    logger.error(f"{'='*60}")
    raise RuntimeError(f"CHECKPOINT FAILED: {name} — {reason}")
