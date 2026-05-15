import cv2
import mediapipe as mp
import math

# --- HELPER FUNCTION: Calculate distance between two 3D landmarks ---
def calculate_distance(point1, point2):
    return math.hypot(point2.x - point1.x, point2.y - point1.y)

# --- HELPER FUNCTION: Calculate Eye Aspect Ratio ---
def calculate_ear(eye_points, landmarks):
    # Map the specific eye IDs to their actual coordinates
    p1 = landmarks.landmark[eye_points[0]]
    p2 = landmarks.landmark[eye_points[1]]
    p3 = landmarks.landmark[eye_points[2]]
    p4 = landmarks.landmark[eye_points[3]]
    p5 = landmarks.landmark[eye_points[4]]
    p6 = landmarks.landmark[eye_points[5]]

    # Calculate height distances (Vertical)
    vertical_1 = calculate_distance(p2, p6)
    vertical_2 = calculate_distance(p3, p5)
    
    # Calculate width distance (Horizontal)
    horizontal = calculate_distance(p1, p4)

    # The EAR Formula
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

# Right eye landmark IDs in MediaPipe Face Mesh
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Alarm variables
EAR_THRESHOLD = 0.25 # If EAR drops below this, eye is closed
DROWSY_FRAMES = 15   # How many consecutive frames the eye must be closed to trigger the alarm
closed_frame_count = 0

with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

                # 1. Calculate the EAR for the right eye
                ear = calculate_ear(RIGHT_EYE, face_landmarks)

                # 2. Draw the EAR value on the screen for debugging
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # 3. Check the Logic
                if ear < EAR_THRESHOLD:
                    # Eye is closed, start counting frames
                    closed_frame_count += 1
                else:
                    # Eye is open, reset the counter
                    closed_frame_count = 0

                # 4. Trigger the Alarm
                if closed_frame_count >= DROWSY_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (30, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    # ---> HARDWARE INTEGRATION POINT <---
                    # This is where you would use pyserial to send a 'HIGH' signal 
                    # to your ESP32 to trigger a physical buzzer or flashing LED.

        cv2.imshow('Drowsiness Detector', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()