import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

from models.vision.vision_model import VisionCNN
from models.audio.audio_model import AudioCNN
from models.fusion.fusion_model import FusionModel

# Dataset to load face image, spectrogram, and motion scalar
class DeepFakeDataset(Dataset):
    def __init__(self, labels_path, frame_dir, spec_dir):
        df = pd.read_csv(labels_path)
        self.items = df.to_dict('records')
        self.frame_dir = frame_dir
        self.spec_dir = spec_dir

        self.frame_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.spec_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_id = item['id']
        label = float(item['label'])
        motion = float(item['motion'])

        frame_path = os.path.join(self.frame_dir, f"{img_id}.jpg")
        spec_path = os.path.join(self.spec_dir, f"{img_id}.png")

        frame = self.frame_tf(Image.open(frame_path).convert("RGB"))
        spec = self.spec_tf(Image.open(spec_path).convert("RGB"))
        motion = torch.tensor(motion, dtype=torch.float32)

        return frame, spec, motion, torch.tensor(label, dtype=torch.float32)

def train():
    # Paths
    LABELS_PATH = "data/labels.csv"
    FRAME_DIR = "data/extracted_frames"
    SPEC_DIR = "data/spectrograms"
    MODEL_OUT = "outputs/deepfake_model.pth"

    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 8
    LR = 1e-4

    # Setup
    dataset = DeepFakeDataset(LABELS_PATH, FRAME_DIR, SPEC_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vision = VisionCNN().to(device)
    audio = AudioCNN().to(device)
    fusion = FusionModel().to(device)

    optimizer = torch.optim.Adam(list(vision.parameters()) +
                                 list(audio.parameters()) +
                                 list(fusion.parameters()), lr=LR)

    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
        vision.train()
        audio.train()
        fusion.train()

        running_loss = 0.0
        for frames, specs, motions, labels in dataloader:
            frames = frames.to(device)
            specs = specs.to(device)
            motions = motions.to(device)
            labels = labels.to(device)

            video_feat = vision(frames)
            audio_feat = audio(specs)
            preds = fusion(video_feat, audio_feat, motions)

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save models
    torch.save({
        'vision': vision.state_dict(),
        'audio': audio.state_dict(),
        'fusion': fusion.state_dict()
    }, MODEL_OUT)
    print(f"[+] Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    train()
