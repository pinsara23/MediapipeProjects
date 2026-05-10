import cv2
import mediapipe as mp

# 1. Initialize the Hands Brain AND the Drawing Utility
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # <-- NEW: The automatic artist that will draw the skeleton for us!

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # If MediaPipe saw a hand...
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # --- NEW CODE STARTS HERE ---
                
                # Tell the drawing utility to draw on the 'frame', 
                # using the data from 'hand_landmarks', 
                # and connect the dots using the standard 'HAND_CONNECTIONS' map.
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
                # --- NEW CODE ENDS HERE ---

        # Show the camera feed (now with a full skeleton!)
        cv2.imshow('Raw Camera Feed', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()