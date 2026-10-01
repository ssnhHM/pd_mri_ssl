import torch.nn as nn


def _group_count(channels, preferred=8):
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_dim=256,
        base_channels=24,
        dropout=0.1,
    ):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]
        blocks = []
        current = in_channels

        for i, output in enumerate(channels):
            blocks.extend(
                [
                    nn.Conv3d(current, output, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(output), output),
                    nn.GELU(),
                    nn.MaxPool3d(2) if i < 3 else nn.Identity(),
                ]
            )
            current = output

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        return self.head(self.pool(x))
