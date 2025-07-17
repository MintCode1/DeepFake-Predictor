import os
import cv2
import dlib
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from PIL import Image

# Define paths
RAW_DIR = "data/raw_videos"
FRAME_DIR = "data/extracted_frames"
AUDIO_DIR = "data/audio"
SPEC_DIR = "data/spectrograms"
LABELS_PATH = "data/labels.csv"
LANDMARK_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# Ensure output dirs exist
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SPEC_DIR, exist_ok=True)

# Load dlib detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL_PATH)

def extract_audio(video_path, output_audio):
    cmd = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{output_audio}\" -y -loglevel error"
    subprocess.call(cmd, shell=True)

def create_spectrogram(audio_path, output_image):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(np.abs(S))

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
    plt.close()

def extract_face_frame(video_path, output_path, motion_log=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = False
    while True:
        ret, frame = cap.read()
        if not ret or count > 50:  # Only check first 50 frames
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            x1, y1, x2, y2 = faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                resized = cv2.resize(cropped, (224, 224))
                cv2.imwrite(output_path, resized)
                success = True
                break
        count += 1
    cap.release()

    # Head motion vector extraction
    if motion_log is not None:
        cap = cv2.VideoCapture(video_path)
        prev_landmarks = None
        deltas = []
        for _ in range(15):  # Use 15 frames for motion tracking
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) == 0:
                continue
            shape = predictor(gray, faces[0])
            landmarks = np.array([[pt.x, pt.y] for pt in shape.parts()])
            if prev_landmarks is not None:
                motion = np.linalg.norm(landmarks - prev_landmarks, axis=1).mean()
                deltas.append(motion)
            prev_landmarks = landmarks
        cap.release()
        avg_motion = np.mean(deltas) if deltas else 0
        motion_log.append(avg_motion)

def get_label_from_filename(filename):
    # Define labeling rule: filename contains 'fake' or 'real'
    if "fake" in filename.lower():
        return 1
    elif "real" in filename.lower():
        return 0
    else:
        raise ValueError(f"Unknown label type for file: {filename}")

def main():
    motion_log = []
    with open(LABELS_PATH, "w") as out_f:
        out_f.write("id,label,motion\n")
        for file in os.listdir(RAW_DIR):
            if not file.endswith(".mp4"):
                continue
            name = os.path.splitext(file)[0]  # e.g. fake_001
            label = get_label_from_filename(name)

            video_path = os.path.join(RAW_DIR, file)
            frame_path = os.path.join(FRAME_DIR, f"{name}.jpg")
            audio_path = os.path.join(AUDIO_DIR, f"{name}.wav")
            spec_path = os.path.join(SPEC_DIR, f"{name}.png")

            try:
                print(f"[+] Processing {file}")
                extract_face_frame(video_path, frame_path, motion_log)
                extract_audio(video_path, audio_path)
                create_spectrogram(audio_path, spec_path)
                avg_motion = motion_log[-1] if motion_log else 0
                out_f.write(f"{name},{label},{avg_motion:.4f}\n")
            except Exception as e:
                print(f"[!] Failed on {file}: {e}")

if __name__ == "__main__":
    main()
