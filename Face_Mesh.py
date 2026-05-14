import cv2
import mediapipe as mp

# 1. Initialize Face Mesh and Drawing Tools
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles # <-- NEW: For the wireframe look

cap = cv2.VideoCapture(0)

# 2. Configure the Face Mesh Brain
# refine_landmarks=True adds 10 extra points specifically for the Irises (eyes)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Process the frame
        results = face_mesh.process(rgb_frame)

        # 4. Extract and Draw
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Draw the Tesselation (The 468-point Wireframe Mask)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None, # Don't draw the dots!
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw the Contours (Thicker lines around the eyes, lips, and face edge)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        cv2.imshow('Face Mesh Liveness Camera', frame)
        
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()