import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import pandas as pd
import numpy as np

from models.vision.vision_model import VisionCNN
from models.audio.audio_model import AudioCNN
from models.fusion.fusion_model import FusionModel

# Modified Dataset with distortion mode
class EvalDataset(Dataset):
    def __init__(self, labels_path, frame_dir, spec_dir, distortion=None):
        df = pd.read_csv(labels_path)
        self.items = df.to_dict('records')
        self.frame_dir = frame_dir
        self.spec_dir = spec_dir
        self.distortion = distortion

        self.frame_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.spec_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def apply_distortion(self, img):
        if self.distortion == "noise":
            np_img = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 15, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(np_img)
        elif self.distortion == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=1.5))
        elif self.distortion == "jpeg":
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=40)
            return Image.open(buf)
        return img  # no distortion

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_id = item['id']
        label = float(item['label'])
        motion = float(item['motion'])

        frame_path = os.path.join(self.frame_dir, f"{img_id}.jpg")
        spec_path = os.path.join(self.spec_dir, f"{img_id}.png")

        frame = Image.open(frame_path).convert("RGB")
        spec = Image.open(spec_path).convert("RGB")

        # Apply distortion
        frame = self.apply_distortion(frame)
        spec = self.apply_distortion(spec)

        frame = self.frame_tf(frame)
        spec = self.spec_tf(spec)
        motion = torch.tensor(motion, dtype=torch.float32)

        return frame, spec, motion, torch.tensor(label, dtype=torch.float32)

def evaluate(distortion=None):
    # Load model
    checkpoint = torch.load("outputs/deepfake_model.pth", map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision = VisionCNN().to(device)
    audio = AudioCNN().to(device)
    fusion = FusionModel().to(device)

    vision.load_state_dict(checkpoint['vision'])
    audio.load_state_dict(checkpoint['audio'])
    fusion.load_state_dict(checkpoint['fusion'])

    vision.eval()
    audio.eval()
    fusion.eval()

    # Data
    dataset = EvalDataset("data/labels.csv", "data/extracted_frames", "data/spectrograms", distortion)
    loader = DataLoader(dataset, batch_size=8)

    preds, targets = [], []

    with torch.no_grad():
        for frames, specs, motions, labels in loader:
            frames = frames.to(device)
            specs = specs.to(device)
            motions = motions.to(device)
            labels = labels.to(device)

            v_feat = vision(frames)
            a_feat = audio(specs)
            out = fusion(v_feat, a_feat, motions)
            preds += out.cpu().numpy().tolist()
            targets += labels.cpu().numpy().tolist()

    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    targets = [int(t) for t in targets]
    acc = np.mean([p == t for p, t in zip(preds_bin, targets)])

    print(f"[{distortion or 'clean'}] Accuracy: {acc:.4f}")

if __name__ == "__main__":
    for d in [None, "noise", "jpeg", "blur"]:
        evaluate(distortion=d)
