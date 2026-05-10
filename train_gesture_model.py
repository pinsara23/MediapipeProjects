import cv2
import mediapipe as mp
import time
import os

# 1. Create a folder to save the images if it doesn't exist
if not os.path.exists("saved_gestures"):
    os.makedirs("saved_gestures")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 2. Setup variables for the Cooldown Timer
last_save_time = 0
cooldown_seconds = 3
photo_count = 0

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        
        # 3. THE CLEAN COPY: Save a copy of the frame before drawing the skeleton on it
        clean_frame = frame.copy() 

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the skeleton on the main display frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- PEACE SIGN LOGIC ---
                # Index and Middle are UP (Y of tip is less than Y of PIP joint)
                index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
                
                # Ring and Pinky are DOWN (Y of tip is greater than Y of PIP joint)
                ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
                pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

                # 4. THE TRIGGER: If it sees a Peace Sign
                if index_up and middle_up and ring_down and pinky_down:
                    current_time = time.time()
                    
                    # 5. THE TIMER: Check if enough time has passed since the last photo
                    if current_time - last_save_time > cooldown_seconds:
                        photo_count += 1
                        filename = f"saved_gestures/peace_sign_{photo_count}.jpg"
                        
                        # 6. THE SAVE: Use OpenCV to write the clean frame to your hard drive
                        cv2.imwrite(filename, clean_frame)
                        print(f"Photo saved: {filename}")
                        
                        # Reset the timer
                        last_save_time = current_time
                        
                    # UI Feedback on the display window
                    cv2.putText(frame, "TAKING PHOTO!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # Show the display frame with the skeleton and UI text
        cv2.imshow('Gesture Auto-Save Camera', frame)
        
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()