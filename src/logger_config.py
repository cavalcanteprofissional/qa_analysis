"""Configuração de logging estruturado para a pipeline."""
import logging
from datetime import datetime
from pathlib import Path
from logging import Logger


def setup_logging(log_dir: str = "logs") -> Logger:
    """Configura logging com arquivo por execução e console.

    Formato: date | level | module | message
    Retorna o logger raiz configurado.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger = logging.getLogger("qa_pipeline")
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.propagate = False
    return logger
