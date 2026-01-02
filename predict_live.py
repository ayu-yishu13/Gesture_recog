import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque


MODE = input("Select mode (gesture / asl): ").strip().lower()

if MODE == "gesture":
    MODEL_PATH = "models/gesture_model.pkl"
elif MODE == "asl":
    MODEL_PATH = "models/asl_static_model.pkl"
else:
    raise ValueError("Mode must be 'gesture' or 'asl'")




bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
LABELS = bundle["labels"]
MODE_NAME = bundle["mode"]


CONFIDENCE_THRESHOLD = 0.90
SMOOTHING_WINDOW = 10

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)

print(f"\nRunning in {MODE_NAME.upper()} mode")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "Unknown"
    confidence = 0.0

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

        
        probs = model.predict_proba([landmarks])[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction_buffer.append(LABELS[pred_idx])
        else:
            prediction_buffer.append("Unknown")

        
        label = max(set(prediction_buffer), key=prediction_buffer.count)

    
    display_text = f"{label} ({confidence:.2f})"
    color = (0, 0, 255) if label != "Unknown" else (100, 100, 100)

    cv2.putText(
        frame,
        f"Mode: {MODE_NAME.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        display_text,
        (40, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        color,
        3
    )

    cv2.imshow("Gesture / ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
