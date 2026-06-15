"""Hydra CLI entry point for the PROSE pipeline."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .pipeline import Pipeline
from .utils.logging import get_logger

log = get_logger(__name__)


def _load_dotenv() -> None:
    """Load variables from `.env` at the repo root if present (no overwrite)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        log.info("Loaded environment from %s", env_path)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).resolve().parents[2] / "configs"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    Pipeline(cfg).run()


if __name__ == "__main__":
    _load_dotenv()
    main()
else:
    _load_dotenv()
