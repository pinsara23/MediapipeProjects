import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Draw the skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 2. Get specific Y-coordinates for logic
                # Thumb tip (4) and base (2)
                thumb_tip = hand_landmarks.landmark[4].y
                thumb_base = hand_landmarks.landmark[2].y

                # Other finger tips (8, 12, 16, 20) and their joints (6, 10, 14, 18)
                index_folded = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
                middle_folded = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
                ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
                pinky_folded = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

                # 3. DEFINE THE GESTURE
                # If thumb is up AND all other fingers are folded
                if thumb_tip < thumb_base and index_folded and middle_folded and ring_folded and pinky_folded:
                    
                    # --- TRIGGER TASK START ---
                    cv2.putText(frame, "THUMBS UP DETECTED!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    
                    # Example of another task: Print to console
                    print("Task Triggered: Saving data or sending to ESP32...")
                    # --- TRIGGER TASK END ---

        cv2.imshow('Gesture Task Trigger', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()