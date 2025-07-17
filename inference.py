import os
import torch
import argparse
from PIL import Image
from torchvision import transforms

from models.vision.vision_model import VisionCNN
from models.audio.audio_model import AudioCNN
from models.fusion.fusion_model import FusionModel

from utils.video_utils import extract_faces
from utils.audio_utils import extract_audio, create_spectrogram
from utils.motion_utils import extract_head_motion

def preprocess(video_path, tmp_prefix="temp"):
    os.makedirs("temp", exist_ok=True)
    face_out = f"temp/{tmp_prefix}.jpg"
    audio_out = f"temp/{tmp_prefix}.wav"
    spec_out = f"temp/{tmp_prefix}.png"

    extract_faces(video_path, "temp", every_n_frames=5)
    extract_audio(video_path, audio_out)
    create_spectrogram(audio_out, spec_out)
    motion_value = extract_head_motion(video_path)

    return face_out, spec_out, motion_value

def predict(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    checkpoint = torch.load("outputs/deepfake_model.pth", map_location=device)
    vision = VisionCNN().to(device)
    audio = AudioCNN().to(device)
    fusion = FusionModel().to(device)

    vision.load_state_dict(checkpoint['vision'])
    audio.load_state_dict(checkpoint['audio'])
    fusion.load_state_dict(checkpoint['fusion'])

    vision.eval()
    audio.eval()
    fusion.eval()

    # Preprocess input video
    face_path, spec_path, motion = preprocess(video_path)

    # Transforms
    frame_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    spec_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    # Load inputs
    frame = frame_tf(Image.open(face_path).convert("RGB")).unsqueeze(0).to(device)
    spec = spec_tf(Image.open(spec_path).convert("RGB")).unsqueeze(0).to(device)
    motion_tensor = torch.tensor([motion], dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        v_feat = vision(frame)
        a_feat = audio(spec)
        pred = fusion(v_feat, a_feat, motion_tensor).item()

    print(f"\nVideo: {video_path}")
    print(f"Head motion score: {motion:.4f}")
    print(f"Prediction: {'DeepFake (1)' if pred > 0.5 else 'Real (0)'}")
    print(f"Confidence: {pred:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video")
    args = parser.parse_args()

    predict(args.video)
