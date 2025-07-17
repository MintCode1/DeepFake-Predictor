import dlib
import cv2
import numpy as np

predictor_path = "shape_predictor_68_face_landmarks.dat"  # Must exist in project root
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def extract_head_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_landmarks = None
    deltas = []

    while True:
        ret, frame = cap.read()
        if not ret or len(deltas) >= 15:
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
    return float(np.mean(deltas)) if deltas else 0.0
