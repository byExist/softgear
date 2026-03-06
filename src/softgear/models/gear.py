from typing import cast

from torch import Tensor, nn


class Gear(nn.Module):
    """One gear in the chain: a stack of Pre-LN Transformer layers.

    The number of layers (num_layers) determines the gear's "resolution" —
    more layers allow finer-grained computation within a single gear.

    With identity_init=True, output projections are zeroed so that
    the gear acts as identity (x + 0 = x), preserving existing
    chain behavior when a new gear is appended.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        identity_init: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )

        if identity_init:
            for module in self.layers:
                layer = cast(nn.TransformerEncoderLayer, module)
                nn.init.zeros_(layer.self_attn.out_proj.weight)
                nn.init.zeros_(layer.self_attn.out_proj.bias)
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
