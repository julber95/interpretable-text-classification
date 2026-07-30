"""Shared helpers for the single-level and multi-level training entry points."""

import logging
import warnings

import torch


def resolve_accelerator() -> str:
    """Test actual CUDA init — is_available() can return True even when init fails."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except RuntimeError:
            pass
    return "cpu"


def suppress_noisy_logs() -> None:
    """Silence third-party warnings/loggers that otherwise clutter training output."""
    warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
