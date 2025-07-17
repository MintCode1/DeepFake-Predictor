import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_dim_video=128, input_dim_audio=128, input_dim_motion=1):
        super(FusionModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim_video + input_dim_audio + input_dim_motion, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, video_feat, audio_feat, motion_feat):
        if motion_feat.dim() == 1:
            motion_feat = motion_feat.unsqueeze(1)  # Ensure shape [B, 1]
        x = torch.cat([video_feat, audio_feat, motion_feat], dim=1)
        out = self.classifier(x)
        return out.squeeze(1)  # Return [B] for BCELoss
