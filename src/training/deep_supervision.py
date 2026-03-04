import torch
from torch import Tensor, nn

from src.models.base import ModelOutput


class DeepSupervisionLoss(nn.Module):
    """Loss with auxiliary supervision on intermediate round outputs.

    total_loss = main_loss + alpha * sum(intermediate_losses)
    """

    def __init__(self, base_loss_fn: nn.Module, alpha: float = 0.3):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.alpha = alpha

    def _compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.base_loss_fn(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    def forward(self, model_output: ModelOutput, targets: Tensor) -> Tensor:
        main_loss = self._compute_loss(model_output.logits, targets)

        intermediates = model_output.intermediate_logits
        if not intermediates or self.alpha == 0.0:
            return main_loss

        aux_loss = torch.stack(
            [self._compute_loss(inter, targets) for inter in intermediates]
        ).sum()
        return main_loss + self.alpha * aux_loss
