from torch import nn
from loralib import layers as lora_ft

class MLPProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class LoRAMLPProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout, r=4):
        super().__init__()
        self.projection = lora_ft.Linear(embedding_dim, projection_dim, r=r)
        self.gelu = nn.GELU()
        self.fc = lora_ft.Linear(projection_dim, projection_dim, r=r)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class LinearProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projection(x)

class LoRALinearProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, r=4):
        super().__init__()
        self.projection = lora_ft.Linear(embedding_dim, projection_dim, r=r)

    def forward(self, x):
        return self.projection(x)
