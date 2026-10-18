"""
util.py  --  Logging, output-folder, and evaluation-grid utilities.
"""

import os
import sys
import time
import logging
import platform

import pytz
import torch


def create_output_folder(parent_folder: str = "out", prefix: str = "",
                         postfix: str = "") -> str:
    """
    Create a timestamped output folder (Berlin time zone).

    Returns the absolute path of the created folder.
    """
    time_stamp = int(time.time())
    time_zone = pytz.timezone("Europe/Berlin")
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(os.getcwd(), parent_folder,
                          f"{prefix}{test_time}{postfix}")
    os.makedirs(folder, exist_ok=True)
    return folder


def setup_logger(save_folder: str, name: str = "train") -> logging.Logger:
    """
    Create a logger that writes to both console and a log file.

    The log file is saved as ``{save_folder}/{name}.log``.

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(message)s")

    # File handler
    log_path = os.path.join(save_folder, f"{name}.log")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def log_run_header(logger: logging.Logger, args, net=None,
                   family=None) -> None:
    """
    Log device info, hyperparameters, and network summary.

    Call this at the start of training, after building the model.
    """
    tz = pytz.timezone("Europe/Berlin")
    now = pytz.datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    logger.info("=" * 60)
    logger.info("Two-Density Fluid Mixing Operator -- Training")
    logger.info("=" * 60)
    logger.info(f"  Start time       : {now}")
    logger.info(f"  Python           : {platform.python_version()}")
    logger.info(f"  PyTorch          : {torch.__version__}")
    logger.info(f"  Device           : {args.device}")

    if args.device != "cpu" and torch.cuda.is_available():
        logger.info(f"  GPU              : {torch.cuda.get_device_name(args.device)}")
        mem_gb = torch.cuda.get_device_properties(args.device).total_memory / 1e9
        logger.info(f"  GPU memory       : {mem_gb:.1f} GB")

    logger.info(f"  CUDA available   : {torch.cuda.is_available()}")
    logger.info("-" * 60)
    logger.info("  Hyperparameters:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"    {k:<16} = {v}")

    if family is not None:
        logger.info("-" * 60)
        logger.info("  Partition family:")
        info = family.info() if hasattr(family, "info") else {}
        logger.info(f"    name             = {info.get('name', '?')}")
        logger.info(f"    dim              = {info.get('dim', '?')}")
        logger.info(f"    bounds           = {info.get('bounds', '?')}")
        logger.info(f"    hole             = {info.get('hole', '?')}")

    if net is not None:
        logger.info("-" * 60)
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.info(f"  Network parameters : {n_params:,}")

    logger.info("=" * 60)


def log_training_summary(logger: logging.Logger, start_time: float,
                         n_iters: int, save_path: str) -> None:
    """Log a summary block at the end of training."""
    total_time = time.time() - start_time
    tz = pytz.timezone("Europe/Berlin")
    now = pytz.datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

    hours, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"  End time         : {now}")
    logger.info(f"  Total wall time  : {int(hours)}h {int(mins)}m {secs:.1f}s")
    logger.info(f"  Iterations       : {n_iters}")
    if n_iters > 0:
        logger.info(f"  Time per iter    : {total_time / n_iters * 1000:.1f} ms")
    logger.info(f"  Model saved to   : {save_path}")
    logger.info("=" * 60)


def make_lattice(N_grid: int, device: str = 'cpu') -> torch.Tensor:
    """
    Create a uniform lattice of N_grid x N_grid particles on [0,1]^2.

    Returns X : (1, N_grid^2, 2)
    """
    g = torch.linspace(0.5 / N_grid, 1.0 - 0.5 / N_grid, N_grid, device=device)
    x1, x2 = torch.meshgrid(g, g, indexing='ij')
    X = torch.stack([x1.reshape(-1), x2.reshape(-1)], dim=-1)
    return X.unsqueeze(0)
