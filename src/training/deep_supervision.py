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

    def _compute_loss(
        self, logits: Tensor, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Compute loss, optionally restricted to positions where mask is True."""
        if mask is not None:
            logits = logits[mask]
            targets = targets[mask]
        return self.base_loss_fn(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    def forward(
        self, model_output: ModelOutput, targets: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        main_loss = self._compute_loss(model_output.logits, targets, mask)

        intermediates = model_output.intermediate_logits
        if not intermediates or self.alpha == 0.0:
            return main_loss

        aux_loss = torch.stack(
            [self._compute_loss(inter, targets, mask) for inter in intermediates]
        ).sum()
        return main_loss + self.alpha * aux_loss
