import cv2
import pickle
import mediapipe as mp
import numpy as np
import streamlit as st
from collections import deque, Counter

st.set_page_config(
    page_title="ASL Sign Language Detector",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>
.letter-box {
    font-size: 100px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 16px;
    background: #1a1a2e;
    color: #00ff88;
    border: 2px solid #00ff88;
    margin-bottom: 12px;
}
.sentence-box {
    font-size: 28px;
    text-align: center;
    padding: 16px;
    border-radius: 12px;
    background: #1a1a2e;
    color: white;
    border: 1px solid #333;
    min-height: 60px;
    letter-spacing: 6px;
    margin-bottom: 12px;
}
.conf-text {
    text-align: center;
    font-size: 16px;
    color: #aaa;
    margin-bottom: 8px;
}
.tip {
    padding: 8px 12px;
    background: #1a1a2e;
    border-left: 3px solid #00ff88;
    border-radius: 0 8px 8px 0;
    color: #ccc;
    font-size: 13px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open("models/sign_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["encoder"]

@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8
    )
    return hands, mp_hands

def get_landmarks(landmarks):
    lm = np.array([[p.x, p.y, p.z] for p in landmarks])
    lm -= lm[0]
    scale = np.linalg.norm(lm[9]) + 1e-6
    lm /= scale
    return lm.flatten()

# Load resources
try:
    model, le = load_model()
    hands, mp_hands = load_mediapipe()
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"Could not load model: {e}. Please run train.py first.")
    st.stop()

# Session state
if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "stable_letter" not in st.session_state:
    st.session_state.stable_letter = "—"
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=10)
if "total" not in st.session_state:
    st.session_state.total = 0

# Title
st.markdown("## 🤟 ASL Sign Language Detector")
st.markdown("Detect all 26 letters of the American Sign Language alphabet in real time.")
st.divider()

# Layout
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 📷 Camera feed")
    camera_frame = st.empty()

with right:
    st.markdown("### Detected letter")
    letter_display = st.empty()
    letter_display.markdown('<div class="letter-box">—</div>', unsafe_allow_html=True)

    st.markdown("### Confidence")
    conf_display = st.empty()
    conf_display.markdown('<div class="conf-text">0%</div>', unsafe_allow_html=True)
    conf_progress = st.progress(0)

    st.markdown("### Sentence")
    sentence_display = st.empty()
    sentence_display.markdown('<div class="sentence-box">_</div>', unsafe_allow_html=True)

    # Buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Add letter", use_container_width=True):
            if st.session_state.stable_letter not in ["—", ""]:
                st.session_state.sentence.append(st.session_state.stable_letter)
                st.session_state.total += 1
    with c2:
        if st.button("⎵ Add space", use_container_width=True):
            st.session_state.sentence.append(" ")
    with c3:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.sentence = []

    st.divider()

    # Stats
    st.markdown("### Stats")
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Letters in model", len(le.classes_))
    with s2:
        st.metric("Letters added", st.session_state.total)

    st.divider()

    # Tips
    st.markdown("### Tips")
    st.markdown('<div class="tip">Keep hand inside camera frame</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip">All fingers must be visible</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip">Good lighting on your hand</div>', unsafe_allow_html=True)
    st.markdown('<div class="tip">Hold sign steady for 1–2 seconds</div>', unsafe_allow_html=True)

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    st.error("Camera not found! Make sure webcam is connected.")
    st.stop()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    letter = "—"
    confidence = 0.0

    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        features = get_landmarks(result.multi_hand_landmarks[0].landmark)
        probs = model.predict_proba([features])[0]
        idx = np.argmax(probs)
        confidence = float(probs[idx])
        letter = le.inverse_transform([idx])[0]
        st.session_state.buffer.append(letter)

        if len(st.session_state.buffer) == 10:
            most_common, freq = Counter(st.session_state.buffer).most_common(1)[0]
            if freq >= 7:
                st.session_state.stable_letter = most_common

    # Draw on frame
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
    color = (0, 255, 136) if letter != "—" else (100, 100, 100)
    cv2.putText(frame, f"{st.session_state.stable_letter}  {confidence:.0%}",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2)

    # Update UI
    camera_frame.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True
    )

    letter_display.markdown(
        f'<div class="letter-box">{st.session_state.stable_letter}</div>',
        unsafe_allow_html=True
    )

    conf_display.markdown(
        f'<div class="conf-text">{confidence:.0%}</div>',
        unsafe_allow_html=True
    )
    conf_progress.progress(int(confidence * 100))

    sentence_text = "".join(st.session_state.sentence) if st.session_state.sentence else "_"
    sentence_display.markdown(
        f'<div class="sentence-box">{sentence_text}</div>',
        unsafe_allow_html=True
    )

cap.release()