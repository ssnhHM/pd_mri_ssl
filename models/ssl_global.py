import torch.nn as nn

from models.global_encoder import GlobalEncoder
from models.projection_heads import ProjectionHead


class GlobalSSL(nn.Module):
    def __init__(
        self,
        in_channels,
        embedding_dim=256,
        projection_dim=128,
        base_channels=24,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = GlobalEncoder(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            dropout=dropout,
        )
        self.projector = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim * 2,
            output_dim=projection_dim,
            dropout=dropout,
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        embedding = self.encode(x)
        projection = self.projector(embedding)
        return {"z_global": embedding, "p_global": projection}
