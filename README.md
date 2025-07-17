# DeepFake Analysis and Predictor

## Overview:

A complete end-to-end pipeline for detecting DeepFakes using:

- Convolutional Neural Networks (CNNs) for face image artifacts

- Frequency-domain analysis (FFT + spectrograms) for synthetic voice patterns

- Head motion vector extraction for physical consistency detection

- Robust evaluation under adversarial perturbations (noise, compression, blur)

- A multimodal fusion classifier combining all signals

## Project Directory Structure

deepfake_predictor/
├── data/
│   ├── raw_videos/            # Input .mp4 videos (real/fake)
│   ├── extracted_frames/      # Cropped face images
│   ├── audio/                 # Extracted audio (.wav)
│   ├── spectrograms/          # Spectrogram images (.png)
│   └── labels.csv             # id,label,motion format
│
├── scripts/
│   └── extract_and_label.py   # Preprocess and generate labels
├── train.py                   # Train multimodal detector
├── eval.py                    # Evaluate on clean + adversarial distortions
├── inference.py               # Predict real/fake on new video
│
├── models/
│   ├── vision/vision_model.py
│   ├── audio/audio_model.py
│   └── fusion/fusion_model.py
│
├── utils/
│   ├── video_utils.py
│   ├── audio_utils.py
│   └── motion_utils.py
│
├── requirements.txt
└── README.md

