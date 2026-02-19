"""
ASL Data Collector
====================
Use your webcam to record hand landmark samples for each letter.
Saves to data/landmarks.csv which is used by train_model.py.

Controls:
  - SPACE  to toggle auto-capture ON/OFF
  - N      to advance to the next letter
  - ENTER  for a single manual capture
  - Q      to quit and save

Usage:
    python collect_data.py --samples 100
"""

import cv2
import csv
import os
import time
import argparse
import numpy as np
import mediapipe as mp
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int,  default=100,             help="Target samples per letter")
parser.add_argument("--output",  default="data/landmarks.csv",       help="Output CSV path")
parser.add_argument("--camera",  type=int,  default=0,               help="Camera index")
parser.add_argument("--delay",   type=float, default=0.1,            help="Delay between auto-captures (seconds)")
args = parser.parse_args()

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 ASL letters (no J/Z)

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.6
)

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
    return arr.tolist()

def main():
    os.makedirs("data", exist_ok=True)

    # Load existing counts if resuming
    counts = defaultdict(int)
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for row in csv.reader(f):
                if row:
                    counts[row[0]] += 1
        print(f"[INFO] Resuming. Current counts: {dict(counts)}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}. Try --camera 1")
        return

    current_label_idx = 0
    auto_capture      = False
    last_capture_time = 0
    status_msg        = "Press SPACE to start capturing!"
    status_time       = time.time()

    csv_file = open(args.output, "a", newline="")
    writer   = csv.writer(csv_file)

    print("[INFO] Webcam open! Press SPACE to start auto-capture. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_detected = False
        features      = None

        if results.multi_hand_landmarks:
            hand_detected = True
            for hlm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                features = extract_landmarks(hlm)

        current_label = LABELS[current_label_idx]
        count         = counts[current_label]
        now           = time.time()

        # Auto-capture logic
        if auto_capture and hand_detected and features:
            if now - last_capture_time >= args.delay:
                if count < args.samples:
                    writer.writerow([current_label] + features)
                    counts[current_label] += 1
                    count += 1
                    last_capture_time = now
                    status_msg  = f"Captured! {count}/{args.samples}"
                    status_time = now
                else:
                    auto_capture = False
                    status_msg   = f"{current_label} DONE! Press N for next letter."
                    status_time  = now

        # ── UI ──────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(frame, "ASL Data Collector", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        mode_col = (0, 220, 80) if auto_capture else (100, 100, 100)
        cv2.putText(frame, "AUTO ON" if auto_capture else "AUTO OFF",
                    (w - 160, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_col, 2)

        # Current letter box
        box_col = (0, 160, 80) if auto_capture else (60, 60, 60)
        cv2.rectangle(frame, (15, 90), (200, 250), box_col, -1)
        cv2.putText(frame, current_label, (40, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 6.0, (255, 255, 255), 8)

        # Progress bar
        progress = min(count / args.samples, 1.0)
        cv2.rectangle(frame, (15, 260), (200, 285), (50, 50, 50), -1)
        cv2.rectangle(frame, (15, 260), (15 + int(185 * progress), 285), (0, 200, 100), -1)
        cv2.putText(frame, f"{count}/{args.samples}", (210, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        # All letters progress grid
        for i, lbl in enumerate(LABELS):
            col       = i % 6
            row       = i // 6
            bx        = 15 + col * 55
            by        = 305 + row * 55
            c         = counts[lbl]
            done      = c >= args.samples
            is_curr   = lbl == current_label
            box_color = (0, 140, 60) if done else (60, 60, 180) if is_curr else (40, 40, 40)
            cv2.rectangle(frame, (bx, by), (bx + 48, by + 48), box_color, -1)
            cv2.putText(frame, lbl, (bx + 8, by + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Status message
        if now - status_time < 3.0:
            cv2.putText(frame, status_msg, (15, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2)

        # Hand indicator
        hand_col = (0, 220, 80) if hand_detected else (0, 80, 220)
        cv2.putText(frame, "Hand: YES" if hand_detected else "Hand: NO",
                    (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hand_col, 2)

        # Controls
        cv2.putText(frame, "[SPACE] Toggle auto  [ENTER] Single  [N] Next letter  [Q] Quit",
                    (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            auto_capture = not auto_capture
            status_msg   = "Auto-capture ON — hold your sign steady!" if auto_capture else "Auto-capture OFF"
            status_time  = now
        elif key == ord('n'):
            auto_capture      = False
            current_label_idx = (current_label_idx + 1) % len(LABELS)
            status_msg        = f"Now recording: {LABELS[current_label_idx]}"
            status_time       = now
        elif key == 13 and hand_detected and features:
            writer.writerow([current_label] + features)
            counts[current_label] += 1
            status_msg  = f"Saved {current_label}! ({counts[current_label]}/{args.samples})"
            status_time = now
        elif chr(key).upper() in LABELS:
            current_label_idx = LABELS.index(chr(key).upper())
            status_msg        = f"Now recording: {LABELS[current_label_idx]}"
            status_time       = now

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    total = sum(counts.values())
    print(f"\n[INFO] Done! Total samples saved: {total}")
    print(f"[INFO] Saved to: {args.output}")
    print("\nPer-letter summary:")
    for lbl in LABELS:
        bar = "█" * min(counts[lbl] // 5, 20)
        print(f"  {lbl}: {counts[lbl]:>4}/{args.samples}  {bar}")

if __name__ == "__main__":
    main()
