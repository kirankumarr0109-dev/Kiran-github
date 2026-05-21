import cv2
import os
import csv
import mediapipe as mp
import numpy as np

DATA_DIR = "data/raw"
OUTPUT_CSV = "data/processed/landmarks.csv"
os.makedirs("data/processed", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1
)

def get_landmarks(landmarks):
    lm = np.array([[p.x, p.y, p.z] for p in landmarks])
    # Normalize: center at wrist, scale by hand size
    lm -= lm[0]
    scale = np.linalg.norm(lm[9]) + 1e-6
    lm /= scale
    return lm.flatten().tolist()

rows = []
skipped = 0
total = sum(
    len(os.listdir(f"{DATA_DIR}/{s}"))
    for s in os.listdir(DATA_DIR)
    if os.path.isdir(f"{DATA_DIR}/{s}")
)
processed = 0

print(f"Processing {total} images...")

for sign in sorted(os.listdir(DATA_DIR)):
    sign_path = f"{DATA_DIR}/{sign}"
    if not os.path.isdir(sign_path):
        continue
    sign_count = 0
    for img_file in os.listdir(sign_path):
        img_path = f"{sign_path}/{img_file}"
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = get_landmarks(result.multi_hand_landmarks[0].landmark)
            rows.append(lm + [sign])
            sign_count += 1
        else:
            skipped += 1
        processed += 1
        if processed % 200 == 0:
            pct = processed * 100 // total
            print(f"Progress: {processed}/{total} ({pct}%) | {sign}: {sign_count} detected")

    print(f"[{sign}] {sign_count} landmarks saved out of 300")

# Write CSV
cols = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")] + ["label"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(cols)
    writer.writerows(rows)

print(f"\nTotal saved: {len(rows)}")
print(f"Skipped (no hand): {skipped}")
print("\nNow run: python train.py")