import logging

import hydra
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from src.data.sudoku import build_sudoku_loaders
    from src.evaluation.sudoku_metrics import sudoku_accuracy
    from src.tasks.sudoku import build_sudoku_model, mount_all_gears
    from src.utils.device import get_device

    device = get_device()

    model = build_sudoku_model(cfg.model)
    mount_all_gears(model, cfg.model)
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    _, val_loader = build_sudoku_loaders(cfg)

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)

            output = model(inputs)
            preds = output.logits.argmax(dim=-1)

            all_preds.append(preds)
            all_targets.append(targets)
            all_inputs.append(inputs)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    inputs = torch.cat(all_inputs)

    metrics = sudoku_accuracy(preds, targets, inputs)
    log.info("Cell accuracy:   %.4f", metrics["cell_accuracy"])
    log.info("Blank accuracy:  %.4f", metrics["blank_accuracy"])
    log.info("Puzzle accuracy: %.4f", metrics["puzzle_accuracy"])


if __name__ == "__main__":
    main()
