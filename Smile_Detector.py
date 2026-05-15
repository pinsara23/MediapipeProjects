import cv2
import mediapipe as mp
import math

# --- HELPER FUNCTION: Calculate distance between two 3D landmarks ---
def calculate_distance(point1, point2):
    return math.hypot(point2.x - point1.x, point2.y - point1.y)

# --- HELPER FUNCTION: Calculate Smile Ratio ---
def calculate_smile_ratio(landmarks):
    # Map the specific eye IDs to their actual coordinates
    p1 = landmarks.landmark[61]
    p2 = landmarks.landmark[291]
    p3 = landmarks.landmark[234]
    p4 = landmarks.landmark[454]
    
    #Mouth Width
    mw = calculate_distance(p1, p2)

    #Face width
    fw = calculate_distance(p3, p4)
    
    ratio = mw/fw

    return ratio

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

# Alarm variables
SMILE_THRESHOLD = 0.40 # If smile ratio exceeds this, smile is detected
SMILE_FRAMES = 10    # How many consecutive frames the smile must be detected to trigger the alarm
smile_frame_count = 0

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

                # 1. Calculate the smile ratio
                smile_ratio = calculate_smile_ratio(face_landmarks)

                # 2. Draw the smile ratio on the screen for debugging
                cv2.putText(frame, f"Smile Ratio: {smile_ratio:.2f}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            
                # Trigger the Alarm
                if smile_ratio >= SMILE_THRESHOLD:
                    cv2.putText(frame, "SMILE DETECTED!", (30, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                    
                    # ---> HARDWARE INTEGRATION POINT <---
                    # This is where you would use pyserial to send a 'HIGH' signal 
                    # to your ESP32 to trigger a physical buzzer or flashing LED.

        cv2.imshow('Smile Detector', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()