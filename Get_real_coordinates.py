import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Grab Point 8 (The Index Finger Tip)
                index_tip = hand_landmarks.landmark[8]
                
                # --- NEW CODE STARTS HERE ---
                
                # 1. Get the exact height and width of your video frame
                h, w, c = frame.shape
                
                # 2. Convert the 0.0-1.0 math into exact screen pixels
                cx = int(index_tip.x * w)
                cy = int(index_tip.y * h)
                
                # 3. Draw a blue circle on the frame at those exact coordinates
                # cv2.circle(image, (X, Y), radius, (Blue, Green, Red), thickness)
                cv2.circle(frame, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

                print(f"Index Tip -> X: {cx:.2f} | Y: {cy:.2f}")
                
                # --- NEW CODE ENDS HERE ---

        # Show the camera feed (now with a blue dot!)
        cv2.imshow('Raw Camera Feed', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()