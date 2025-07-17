import subprocess
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def extract_audio(video_path, audio_out_path):
    command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_out_path}\" -y -loglevel error"
    subprocess.call(command, shell=True)

def create_spectrogram(audio_path, output_image_path):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(abs(S))

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
