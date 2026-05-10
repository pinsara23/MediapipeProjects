import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 1. Extract coordinates for Thumb Tip (4) and Index Tip (8)
                h, w, _ = frame.shape
                x1, y1 = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
                x2, y2 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

                # 2. Draw a line between the two fingers to visualize the distance
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # 3. Calculate the mathematical distance (the hypotenuse)
                distance = math.hypot(x2 - x1, y2 - y1) # distance between 2 points = sqrt((x2-x1)^2 + (y2-y1)^2)  

                # 4. MAP THE DISTANCE TO A TASK
                # Let's assume fingers pinched = 20px, fingers wide open = 200px
                # We want to map that to a Volume Bar from 0 to 100
                
                # np.interp(value_to_check, [raw_min, raw_max], [target_min, target_max])
                vol_percentage = np.interp(distance, [20, 200], [0, 100])
                
                # 5. DRAW THE UI FOR THE TASK
                # Draw a volume bar on the screen
                cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3) # Outline
                
                # Convert percentage to pixel height for the filled bar
                bar_height = int(np.interp(vol_percentage, [0, 100], [400, 150]))
                cv2.rectangle(frame, (50, bar_height), (85, 400), (0, 255, 0), cv2.FILLED) # Fill
                
                # Show the exact percentage text
                cv2.putText(frame, f"{int(vol_percentage)} %", (40, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow('Distance Tracker & Task Trigger', frame)
        
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()