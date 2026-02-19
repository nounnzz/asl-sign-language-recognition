"""
Real-Time ASL Recognition (scikit-learn version)
==================================================
Uses MediaPipe Hands for landmark detection and a trained
scikit-learn model for classifying ASL hand signs A-Y.

Usage:
    python asl_recognition.py
    python asl_recognition.py --model model/asl_model.pkl
    python asl_recognition.py --confidence 0.85
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
import time
import pickle
from collections import deque, Counter

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="model/asl_model.pkl", help="Path to trained model")
parser.add_argument("--confidence", type=float, default=0.75,      help="Minimum confidence (0-1)")
parser.add_argument("--camera",     type=int,   default=0,         help="Camera index")
parser.add_argument("--smooth",     type=int,   default=7,         help="Smoothing window size")
args = parser.parse_args()

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model(path):
    if not os.path.exists(path):
        print(f"[ERROR] Model not found at: {path}")
        print("[INFO] Run 'python train_model.py --data data/landmarks.csv' first.")
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Model loaded from: {path}")
    return data["pipeline"], data["label_encoder"]

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
)

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_landmarks(hand_lm):
    lm_list = []
    for lm in hand_lm.landmark:
        lm_list.extend([lm.x, lm.y, lm.z])
    wrist = lm_list[:3]
    norm = []
    for i in range(0, len(lm_list), 3):
        norm.extend([
            lm_list[i]   - wrist[0],
            lm_list[i+1] - wrist[1],
            lm_list[i+2] - wrist[2],
        ])
    arr = np.array(norm)
    arr /= (np.max(np.abs(arr)) + 1e-6)
    return arr.reshape(1, -1)

# ── Smoothing ─────────────────────────────────────────────────────────────────
class SmoothPredictor:
    def __init__(self, window=7):
        self.window = deque(maxlen=window)

    def update(self, letter):
        self.window.append(letter)

    def get(self):
        if not self.window:
            return None
        counts = Counter(self.window)
        top, count = counts.most_common(1)[0]
        return top if (top is not None and count >= len(self.window) // 2 + 1) else None

# ── UI Helpers ────────────────────────────────────────────────────────────────
def draw_confidence_bar(frame, x, y, w, h, confidence, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    filled = int(w * confidence)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)

def draw_ui(frame, letter, confidence, smoothed, word_buffer, fps, hand_detected):
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.putText(frame, "ASL Real-Time Recognition", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    fps_col = (0, 220, 100) if fps >= 20 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, fps_col, 2)

    # Prediction panel (bottom left)
    px, py = 15, h - 220
    cv2.rectangle(frame, (px, py), (px + 270, h - 15), (20, 20, 20), -1)
    cv2.rectangle(frame, (px, py), (px + 270, h - 15), (80, 80, 80), 1)

    if hand_detected and letter:
        color = (0, 220, 100) if confidence >= args.confidence else (0, 165, 255)
        # Big letter
        cv2.putText(frame, letter, (px + 15, py + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, color, 7)
        # Confidence text + bar
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                    (px + 10, py + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        draw_confidence_bar(frame, px + 10, py + 140, 248, 12, confidence, color)
        # Stable letter
        stable_col = (100, 220, 255) if smoothed else (100, 100, 100)
        cv2.putText(frame, f"Stable: {smoothed or '-'}",
                    (px + 10, py + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, stable_col, 2)
    else:
        cv2.putText(frame, "Show your", (px + 15, py + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        cv2.putText(frame, "hand!", (px + 15, py + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    # Word buffer (bottom right)
    bx = w - 290
    cv2.rectangle(frame, (bx, h - 85), (w - 15, h - 15), (20, 20, 20), -1)
    cv2.rectangle(frame, (bx, h - 85), (w - 15, h - 15), (80, 80, 80), 1)
    cv2.putText(frame, "Word:", (bx + 10, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    display = word_buffer[-14:] if len(word_buffer) > 14 else word_buffer
    cv2.putText(frame, display if display else "_",
                (bx + 10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 80), 2)

    # Controls hint
    cv2.putText(frame, "[SPACE] Add letter  [ENTER] Space  [BKSP] Delete  [C] Clear  [Q] Quit",
                (15, h - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

# ── Main Loop ─────────────────────────────────────────────────────────────────
def main():
    pipeline, le = load_model(args.model)
    if pipeline is None:
        return

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}. Try --camera 1")
        return

    print("[INFO] Real-time recognition started. Press Q to quit.")
    print(f"[INFO] Confidence threshold: {args.confidence*100:.0f}%")

    smoother    = SmoothPredictor(window=args.smooth)
    word_buffer = ""
    fps_hist    = deque(maxlen=30)
    prev_time   = time.time()
    letter      = None
    confidence  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # FPS
        now = time.time()
        fps_hist.append(1.0 / (now - prev_time + 1e-9))
        prev_time = now
        fps = float(np.mean(fps_hist))

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                features = extract_landmarks(hlm)

                # Predict with probability
                proba      = pipeline.predict_proba(features)[0]
                idx        = int(np.argmax(proba))
                confidence = float(proba[idx])
                letter     = le.classes_[idx] if confidence >= args.confidence else "?"

                smoother.update(letter if letter != "?" else None)
        else:
            smoother.update(None)
            letter     = None
            confidence = 0.0

        smoothed = smoother.get()
        draw_ui(frame, letter, confidence, smoothed, word_buffer, fps, hand_detected)
        cv2.imshow("ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' ') and smoothed:
            word_buffer += smoothed
        elif key == 13:  # Enter = add a space between words
            word_buffer += " "
        elif key == 8 and word_buffer:  # Backspace
            word_buffer = word_buffer[:-1]
        elif key == ord('c'):
            word_buffer = ""

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\n[INFO] Session ended. Word built: '{word_buffer}'")

if __name__ == "__main__":
    main()
