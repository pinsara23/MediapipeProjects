import cv2
import mediapipe as mp
import time
import random

COUNTDOWN = 3  # seconds for countdown before showing the gesture

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def identify_gesture(hand_landmarks):
    
    index_tip_y = hand_landmarks.landmark[8].y
    middle_tip_y = hand_landmarks.landmark[12].y
    ring_tip_y = hand_landmarks.landmark[16].y
    pinky_tip_y = hand_landmarks.landmark[20].y
    thumb_tip_y = hand_landmarks.landmark[4].y

    index_base_y = hand_landmarks.landmark[6].y
    middle_base_y = hand_landmarks.landmark[10].y
    ring_base_y = hand_landmarks.landmark[14].y
    pinky_base_y = hand_landmarks.landmark[18].y
    thumb_base_y = hand_landmarks.landmark[2].y

    wrist_y = hand_landmarks.landmark[0].y

    if thumb_tip_y < thumb_base_y and index_tip_y < index_base_y and middle_tip_y < middle_base_y and ring_tip_y < ring_base_y and pinky_tip_y < pinky_base_y:
        return "Paper"
    elif thumb_tip_y < index_base_y and thumb_tip_y < middle_base_y and thumb_tip_y < ring_base_y and thumb_tip_y < pinky_base_y and index_tip_y > index_base_y and middle_tip_y > middle_base_y and ring_tip_y > ring_base_y and pinky_tip_y > pinky_base_y and wrist_y > thumb_tip_y:
        return "Rock"
    elif index_tip_y < index_base_y and middle_tip_y < middle_base_y and ring_tip_y > ring_base_y and pinky_tip_y > pinky_base_y:
        return "Scissors"
    else:
        return "Unknown"
    

computer_choice = random.choice(["Rock", "Paper", "Scissors"])

def getWinner(player, computer):
    if player == computer:
        return "It's a tie!"
    elif (player == "Rock" and computer == "Scissors") or (player == "Paper" and computer == "Rock") or (player == "Scissors" and computer == "Paper"):
        return "You win!"
    else:
        return "Computer wins!"

def show_text(frame,frame_height, frame_width, remaining_time, text):
    
    alpha = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 255, 255)

    # Calculate text size to center it on the screen
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    center_x = (frame_width - text_width) // 2
    center_y = (frame_height - text_height) // 2

    # Draw a semi-transparent rectangle behind the text for better visibility
    '''
    1. Make a copy of the current frame (we'll call it overlay).

    2. Draw your solid rectangle on the overlay.

    3. Use cv2.addWeighted() to blend the overlay and the original frame together.
    '''
    overlay = frame.copy()
    cv2.rectangle(overlay, (center_x - 10, center_y - text_height - 10), (center_x + text_width + 10, center_y + 10), (173, 166, 165), thickness=cv2.FILLED)
    frame = cv2.addWeighted(overlay, alpha ,frame, 1-alpha, 0, frame)

    cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def main():

    cap = cv2.VideoCapture(0)
    # Set the resolution to 960x540 for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    state = "ideal"
    countdown_start = 0
    countdown_value = COUNTDOWN
    computer_choice = "Unknown"
    winner = "UNKNOWN"
    locked_gesture = ""

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
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                )

                h, w, _ = frame.shape

                # Game Logic
                if state == "countdown":
                    elapsed_time = time.time() - countdown_start
                    remaining_time = max(0, int(countdown_value - elapsed_time))

                    text = f"Show your gesture in: {remaining_time}"
                    show_text(frame, h, w, remaining_time, text)

                    if elapsed_time >= countdown_value:
                        locked_gesture = identify_gesture(hand_landmarks)
                        computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                        winner = getWinner(locked_gesture, computer_choice)
                        state = "result"
                        result_time = time.time()

                gesture = identify_gesture(hand_landmarks)
                
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (15, 42, 247), 2)
                cv2.putText(frame, f"Computer: {computer_choice}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (15, 42, 247), 2)

                if state == "result":
                    elapsed_time = time.time() - result_time
                    if elapsed_time < countdown_value:  # Show result for 3 seconds
                        show_text(frame, h, w, remaining_time, f"Winner: {winner}")
                    else:
                        state = "ideal"

        cv2.imshow('Rock Paper Scissors', frame)

        key = cv2.waitKey(5) & 0xFF

        # q to quit, space to start the countdown and lock in the gesture
        if key == ord('q'): 
            break

        if key == ord(' '):
            state = "countdown"
            countdown_start = time.time()
            countdown_value = COUNTDOWN
            computer_choice = "?"
            winner = "UNKNOWN"
            locked_gesture = ""
            

    cap.release()
    cv2.destroyAllWindows()

 
if __name__ == "__main__":
    main()

