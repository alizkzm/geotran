"""Console hygiene: suppress third-party warnings and progress chatter."""

import logging
import os
import warnings

_NOISY_LOGGERS = (
    "datasets", "huggingface_hub", "transformers", "timm",
    "urllib3", "filelock", "fsspec", "PIL", "open_clip",
)


def quiet() -> None:
    """Silence deprecation warnings and Hugging Face / torchvision console noise."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings("ignore")
    for category in (DeprecationWarning, FutureWarning, UserWarning):
        warnings.filterwarnings("ignore", category=category)
    try:
        import numpy as np
        warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
    except (ImportError, AttributeError):
        pass

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        import datasets
        datasets.logging.set_verbosity_error()
        datasets.disable_progress_bars()
    except Exception:
        pass

    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
