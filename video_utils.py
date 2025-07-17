import os
import cv2
import dlib

face_detector = dlib.get_frontal_face_detector()

def extract_faces(video_path, output_dir, every_n_frames=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count > 50:  # limit for efficiency
            break
        if count % every_n_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    output_path = os.path.join(output_dir, f"temp.jpg")
                    resized = cv2.resize(face_img, (224, 224))
                    cv2.imwrite(output_path, resized)
                    cap.release()
                    return output_path
        count += 1
    cap.release()
    return None
