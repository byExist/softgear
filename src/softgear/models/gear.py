from torch import nn


class Gear(nn.TransformerEncoderLayer):
    """Single Pre-LN Transformer layer (= one gear).

    With identity_init=True, output projections are zeroed so that
    the layer acts as identity (x + 0 = x), preserving existing
    block behavior when a new gear is appended.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        identity_init: bool = False,
    ):
        super().__init__(  # type: ignore[reportCallIssue]
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        if identity_init:
            nn.init.zeros_(self.self_attn.out_proj.weight)
            nn.init.zeros_(self.self_attn.out_proj.bias)
            nn.init.zeros_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
