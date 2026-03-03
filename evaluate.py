import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO: Phase 2에서 구현
    # model = load_model(cfg.model, checkpoint_path)
    # dataloader = build_dataloader(cfg.data)
    # metrics = evaluate(model, dataloader)
    print(f"Config loaded: {cfg}")


if __name__ == "__main__":
    main()
