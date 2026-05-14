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

                h, w, _ = frame.shape

                index_tip_sign_y = hand_landmarks.landmark[8].y
                middle_tip_sign_y = hand_landmarks.landmark[12].y
                ring_tip_sign_y = hand_landmarks.landmark[16].y
                pinky_tip_sign_y = hand_landmarks.landmark[20].y
                thumb_tip_sign_y = hand_landmarks.landmark[4].y

                index_base_sign_y = hand_landmarks.landmark[6].y
                middle_base_sign_y = hand_landmarks.landmark[10].y
                ring_base_sign_y = hand_landmarks.landmark[14].y
                pinky_base_sign_y = hand_landmarks.landmark[18].y
                thumb_base_sign_y = hand_landmarks.landmark[3].y

                index_tip_sign_x = hand_landmarks.landmark[8].x
                middle_tip_sign_x = hand_landmarks.landmark[12].x
                ring_tip_sign_x = hand_landmarks.landmark[16].x
                pinky_tip_sign_x = hand_landmarks.landmark[20].x
                thumb_tip_sign_x = hand_landmarks.landmark[4].x

                index_base_sign_x = hand_landmarks.landmark[6].x
                middle_base_sign_x = hand_landmarks.landmark[10].x
                ring_base_sign_x = hand_landmarks.landmark[14].x
                pinky_base_sign_x = hand_landmarks.landmark[18].x
                thumb_base_sign_x = hand_landmarks.landmark[3].x

                wrist_sign_y = hand_landmarks.landmark[0].y

                fly_upwards = thumb_tip_sign_y < thumb_base_sign_y and index_tip_sign_y < index_base_sign_y and middle_tip_sign_y < middle_base_sign_y and ring_tip_sign_y < ring_base_sign_y and pinky_tip_sign_y < pinky_base_sign_y and thumb_tip_sign_x > index_tip_sign_x > middle_tip_sign_x > ring_tip_sign_x > pinky_tip_sign_x 

                fly_downwards = thumb_tip_sign_y > thumb_base_sign_y and index_tip_sign_y > index_base_sign_y and middle_tip_sign_y > middle_base_sign_y and ring_tip_sign_y > ring_base_sign_y and pinky_tip_sign_y > pinky_base_sign_y and index_tip_sign_x < middle_tip_sign_x < ring_tip_sign_x < pinky_tip_sign_x
                
                fly_left = thumb_tip_sign_x < thumb_base_sign_x and index_tip_sign_x < index_base_sign_x and middle_tip_sign_x < middle_base_sign_x and ring_tip_sign_x < ring_base_sign_x and pinky_tip_sign_x < pinky_base_sign_x and index_tip_sign_y < middle_tip_sign_y < ring_tip_sign_y < pinky_tip_sign_y
                
                fly_right = thumb_tip_sign_x > thumb_base_sign_x and index_tip_sign_x > index_base_sign_x and middle_tip_sign_x > middle_base_sign_x and ring_tip_sign_x > ring_base_sign_x and pinky_tip_sign_x > pinky_base_sign_x and index_tip_sign_y < middle_tip_sign_y < ring_tip_sign_y < pinky_tip_sign_y
                
                stop = thumb_tip_sign_y < index_base_sign_y and thumb_tip_sign_y < middle_base_sign_y and thumb_tip_sign_y < ring_base_sign_y and thumb_tip_sign_y < pinky_base_sign_y and index_tip_sign_y > index_base_sign_y and middle_tip_sign_y > middle_base_sign_y and ring_tip_sign_y > ring_base_sign_y and pinky_tip_sign_y > pinky_base_sign_y and wrist_sign_y > thumb_tip_sign_y                 
                
                # 3. DEFINE THE GESTURE
                
                if fly_upwards:
                    
                    # --- TRIGGER TASK START ---

                    cv2.line(frame, (int(thumb_tip_sign_x * w), int(thumb_tip_sign_y * h)), (int(thumb_tip_sign_x * w), 0), (255, 0, 0), 2)
                    cv2.line(frame, (int(index_tip_sign_x * w), int(thumb_tip_sign_y * h)), (int(index_tip_sign_x * w), 0), (255, 0, 0), 2)
                    cv2.line(frame, (int(middle_tip_sign_x * w), int(thumb_tip_sign_y * h)), (int(middle_tip_sign_x * w), 0), (255, 0, 0), 2)
                    cv2.line(frame, (int(ring_tip_sign_x * w), int(thumb_tip_sign_y * h)), (int(ring_tip_sign_x * w), 0), (255, 0, 0), 2)
                    cv2.line(frame, (int(pinky_tip_sign_x * w), int(thumb_tip_sign_y * h)), (int(pinky_tip_sign_x * w), 0), (255, 0, 0), 2)

                    cv2.putText(frame, "FLYING UPWARDS!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    
                    # Example of another task: Print to console
                    print("Task Triggered: Saving data or sending to ESP32...")
                    # --- TRIGGER TASK END ---

                if fly_downwards:
                    
                    # --- TRIGGER TASK START ---
                    cv2.putText(frame, "FLYING DOWNWARDS!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    
                    # Example of another task: Print to console
                    print("Task Triggered: Saving data or sending to ESP32...")
                    # --- TRIGGER TASK END ---

                if fly_left:
                    
                    # --- TRIGGER TASK START ---
                    cv2.putText(frame, "FLYING LEFT!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    
                    # Example of another task: Print to console
                    print("Task Triggered: Saving data or sending to ESP32...")
                    # --- TRIGGER TASK END ---

                if fly_right:
                    
                    # --- TRIGGER TASK START ---
                    cv2.putText(frame, "FLYING RIGHT!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    
                    # Example of another task: Print to console
                    print("Task Triggered: Saving data or sending to ESP32...")
                    # --- TRIGGER TASK END ---

                if stop:
                    
                    # --- TRIGGER TASK START ---
                    cv2.putText(frame, "STOP!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                    
                    # Example of another task: Print to console
                    print("Task Triggered: Saving data or sending to ESP32...")
                    # --- TRIGGER TASK END ---

        cv2.imshow('Gesture Task Trigger', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()