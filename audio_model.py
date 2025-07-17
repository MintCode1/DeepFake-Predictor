import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, output_dim=128):
        super(AudioCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 1, 224, 224] → [B, 32, 112, 112]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 56, 56]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 28, 28]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))  # [B, 128, 1, 1]
        )

        self.projection = nn.Sequential(
            nn.Flatten(),              # [B, 128]
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)           # [B, 128, 1, 1]
        x = self.projection(x)        # [B, output_dim]
        return x
