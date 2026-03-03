import hydra
from omegaconf import DictConfig


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
    seed_everything(cfg.seed)
    # TODO: Implement in Phase 3
    # model = build_model(cfg.model)
    # dataloader = build_dataloader(cfg.data)
    # trainer = build_trainer(cfg.training, model, dataloader)
    # trainer.train()
    print(f"Config loaded: {cfg}")


if __name__ == "__main__":
    main()
