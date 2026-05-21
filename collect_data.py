import cv2
import os
import mediapipe as mp

SIGNS = ['S', 'T', 'U', 'Z']
SAVE_DIR = "data/raw"
SAMPLES_PER_CLASS = 300

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

for sign in SIGNS:
    os.makedirs(f"{SAVE_DIR}/{sign}", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sign_idx = 0
count = 0
collecting = False

print("SPACE=Start/Stop | N=Next Letter | Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_sign = SIGNS[sign_idx]
    hand_detected = result.multi_hand_landmarks is not None

    # Draw hand landmarks
    if hand_detected:
        for lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # Save image only when hand detected
    if collecting and hand_detected:
        cv2.imwrite(f"{SAVE_DIR}/{current_sign}/{count}.jpg", frame)
        count += 1

    # Auto next letter
    if count >= SAMPLES_PER_CLASS:
        print(f"[DONE] {current_sign} - {count} samples saved")
        collecting = False
        count = 0
        sign_idx += 1
        if sign_idx >= len(SIGNS):
            print("All 26 letters collected!")
            break

    # Draw guide box
    cv2.rectangle(frame, (w//2-150, h//2-180),
                  (w//2+150, h//2+180), (0, 255, 255), 2)

    # Status bar
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    hand_text = "HAND: YES" if hand_detected else "HAND: NO - move hand!"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)

    if collecting and hand_detected:
        rec_color = (0, 255, 0)
        rec_text = "RECORDING..."
    elif collecting and not hand_detected:
        rec_color = (0, 0, 255)
        rec_text = "NO HAND!"
    else:
        rec_color = (255, 255, 255)
        rec_text = "PAUSED - press SPACE"

    cv2.putText(frame, f"Letter: {current_sign}  [{count}/{SAMPLES_PER_CLASS}]",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, rec_text,
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)
    cv2.putText(frame, hand_text,
                (w-220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_color, 2)

    # Progress bar
    cv2.rectangle(frame, (0, h-8), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h-8),
                  (int(count/SAMPLES_PER_CLASS*w), h), (0, 200, 100), -1)

    cv2.imshow("ASL Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        collecting = not collecting
    elif key == ord("n"):
        print(f"[SKIP] {current_sign} - saved {count} samples")
        collecting = False
        count = 0
        sign_idx += 1
        if sign_idx >= len(SIGNS):
            break
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCollection done! Now run: python preprocess.py")