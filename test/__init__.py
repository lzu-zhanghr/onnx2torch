from pathlib import Path

from loguru import logger


DIR = Path(__file__).parent
DATASETS_DIR = DIR.parent / "dataset"
LOG_FILE = DIR / "test_model.log"
IMG_DIR = DIR.parent / "img"


logger.add(LOG_FILE, encoding="utf-8", enqueue=True)
