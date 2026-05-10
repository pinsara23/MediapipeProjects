import cv2
import mediapipe as mp

# Initialize the MediaPipe Hands "Brain"
mp_hands = mp.solutions.hands

# Start your webcam
cap = cv2.VideoCapture(0)

# Configure the model to look for exactly 1 hand
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # STEP 1: Capture & Correct (Flip like a mirror)
        frame = cv2.flip(frame, 1)

        # STEP 2: The Color Swap (BGR to RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # STEP 3: Process the image
        results = hands.process(rgb_frame)

        # STEP 4: Extract the Data
        # If MediaPipe saw a hand...
        if results.multi_hand_landmarks:
            # Loop through the hands (we restricted it to 1 max anyway)
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Grab exactly Point 8 (The Index Finger Tip)
                index_tip = hand_landmarks.landmark[8]
                
                # Print the raw, normalized math to the terminal (rounded to 2 decimal places)
                print(f"Index Tip -> X: {index_tip.x:.2f} | Y: {index_tip.y:.2f} | Z: {index_tip.z:.2f}")

        # Show the raw camera feed just so we know it's running
        cv2.imshow('Raw Camera Feed', frame)

        # Press 'Esc' to quit
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Clean up when done
cap.release()
cv2.destroyAllWindows()