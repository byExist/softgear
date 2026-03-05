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

    from src.data.sudoku import build_sudoku_loaders
    from src.tasks.sudoku import build_sudoku_model
    from src.training.trainer import SoftGearTrainer
    from src.utils.device import get_device

    seed_everything(cfg.seed)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = get_device()
    model = build_sudoku_model(cfg.model)
    log.info("Model parameters: %d", model.parameter_count())

    train_loader, val_loader = build_sudoku_loaders(cfg)
    log.info("Train: %d batches, Val: %d batches", len(train_loader), len(val_loader))

    trainer = SoftGearTrainer(cfg, model, train_loader, val_loader, device=device)

    checkpoint_dir = cfg.training.get("checkpoint_dir", "checkpoints")
    resume_path = cfg.get("resume", None)
    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train(checkpoint_dir=checkpoint_dir)


if __name__ == "__main__":
    main()
