import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# 1. Allow up to 2 hands
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 2. Check if hands are detected AND we have Left/Right label data
        if results.multi_hand_landmarks and results.multi_handedness:
            
            # 3. Loop through BOTH the coordinates and the Left/Right labels simultaneously
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Extract the raw label MediaPipe gives us
                raw_label = handedness.classification[0].label
                
                # Fix the Mirror Effect so the label matches your physical hand
                hand_label = "Right" if raw_label == "Left" else "Left"

                # Draw the skeleton on whichever hand it just found
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the wrist coordinates to display text neatly next to the hand
                h, w, _ = frame.shape
                wrist_x = int(hand_landmarks.landmark[0].x * w)
                wrist_y = int(hand_landmarks.landmark[0].y * h)

                # --- TASK 1: RIGHT HAND LOGIC (Pinch Distance) ---
                if hand_label == "Right":
                    x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
                    x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Draw Green Line
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    cv2.putText(frame, f"RIGHT TASK: Dial {int(distance)}", (wrist_x - 50, wrist_y + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- TASK 2: LEFT HAND LOGIC (Thumb Up/Down) ---
                elif hand_label == "Left":
                    thumb_tip = hand_landmarks.landmark[4].y
                    thumb_base = hand_landmarks.landmark[2].y
                    
                    if thumb_tip < thumb_base:
                        status = "Switch ON"
                        color = (255, 0, 0) # Blue
                    else:
                        status = "Switch OFF"
                        color = (0, 0, 255) # Red
                        
                    cv2.putText(frame, f"LEFT TASK: {status}", (wrist_x - 50, wrist_y + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Dual Hand Multitasking', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()