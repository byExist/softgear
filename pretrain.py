import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from src.models.softgear import SoftGearModel
    from src.training.trainer import SoftGearTrainer
    from src.utils.device import get_device

    seed_everything(cfg.seed)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = get_device()
    model = SoftGearModel(cfg.model)
    log.info("Model parameters: %d", model.parameter_count())

    # TODO: Replace with real data loaders in Phase 4
    train_loader: DataLoader = DataLoader([])  # type: ignore[arg-type]
    val_loader: DataLoader = DataLoader([])  # type: ignore[arg-type]

    trainer = SoftGearTrainer(cfg, model, train_loader, val_loader, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
