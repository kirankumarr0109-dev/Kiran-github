import cv2
import pickle
import mediapipe as mp
import numpy as np
from collections import deque, Counter

# Load model
with open("models/sign_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
le = data["encoder"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

def get_landmarks(landmarks):
    lm = np.array([[p.x, p.y, p.z] for p in landmarks])
    lm -= lm[0]
    scale = np.linalg.norm(lm[9]) + 1e-6
    lm /= scale
    return lm.flatten()

# Smooth predictions over last 10 frames
buffer = deque(maxlen=10)
sentence = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("SPACE=Add letter | BACKSPACE=Delete | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter = "-"
    confidence = 0.0
    stable_letter = "-"

    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        features = get_landmarks(result.multi_hand_landmarks[0].landmark)
        probs = model.predict_proba([features])[0]
        idx = np.argmax(probs)
        confidence = probs[idx]
        letter = le.inverse_transform([idx])[0]
        buffer.append(letter)

        # Stable prediction — 7 out of 10 frames agree
        if len(buffer) == 10:
            most_common, freq = Counter(buffer).most_common(1)[0]
            if freq >= 7:
                stable_letter = most_common

    # UI
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Sign: {stable_letter}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
    cv2.putText(frame, f"Conf: {confidence:.0%}",
                (w-160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Raw: {letter}",
                (w-160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

    # Sentence bar
    cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "".join(sentence),
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # Confidence bar
    cv2.rectangle(frame, (10, 90), (200, 110), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 90),
                  (10 + int(confidence * 190), 110), (0, 200, 100), -1)
    cv2.putText(frame, "Confidence",
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("ASL Detection - A to Z", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" ") and stable_letter != "-":
        sentence.append(stable_letter)
    elif key == 8 and sentence:
        sentence.pop()

cap.release()
cv2.destroyAllWindows()