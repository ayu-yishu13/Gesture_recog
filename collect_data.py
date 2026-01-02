import cv2
import mediapipe as mp
import numpy as np
import os


MODE = input("Select mode (gesture / asl): ").strip().lower()

if MODE not in ["gesture", "asl"]:
    raise ValueError("Mode must be 'gesture' or 'asl'")

if MODE == "asl":
    LABEL = input("Enter ASL letter (e.g. A / B / C): ").strip()
    
elif MODE == "gesture":
    LABEL = input("Enter Gesture sign (e.g. fist / palm / point): ").strip()
    
if MODE == "gesture":
    LABEL = LABEL.lower()
    SAVE_DIR = f"data/basic_gestures/{LABEL}"
else:  # asl
    LABEL = LABEL.upper()
    SAVE_DIR = f"data/asl_static/{LABEL}"

os.makedirs(SAVE_DIR, exist_ok=True)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
sample_count = 0

print("Press 's' to save sample, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_landmarks = result.multi_hand_landmarks[0]
        handedness = result.multi_handedness[0].classification[0].label

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
        landmarks = []
        wrist = hand_landmarks.landmark[0]

        for lm in hand_landmarks.landmark:
            x = lm.x - wrist.x
            y = lm.y - wrist.y
            z = lm.z - wrist.z

            
            if handedness == "Left":
                x = -x

            landmarks.extend([x, y, z])

    cv2.putText(
        frame,
        f"Mode: {MODE.upper()} | Label: {LABEL} | Samples: {sample_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and result.multi_hand_landmarks:
        np.save(f"{SAVE_DIR}/{sample_count}.npy", np.array(landmarks))
        sample_count += 1
        print("Saved sample", sample_count)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Data collection complete.")

