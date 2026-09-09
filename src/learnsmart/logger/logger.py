import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / 'LOGS'

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"


LOG_FORMAT = (
    "[ %(asctime)s ] %(levelname)s "
    "%(name)s - %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def get_logger(name: str) -> logging.Logger:

    """
    Return a configured lohher for the given module name
    """

    return logging.getLogger(name)