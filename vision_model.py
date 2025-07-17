import torch.nn as nn
import torchvision.models as models

class VisionCNN(nn.Module):
    def __init__(self, output_dim=128):
        super(VisionCNN, self).__init__()

        # Load pretrained ResNet18 backbone
        base_model = models.resnet18(pretrained=True)
        
        # Remove the final classifier layer (fc)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # output: [batch, 512, 1, 1]
        
        # New projection head to reduce to output_dim
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)         # shape: [B, 512, 1, 1]
        x = self.projector(x)        # shape: [B, output_dim]
        return x                     # embedding vector
