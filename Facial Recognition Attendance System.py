import cv2
import numpy as np
import face_recognition

# Load pre-trained face recognition model
known_faces_encodings = []
known_faces_names = []

# Load known faces and names (replace with your dataset)
# For example:
# known_faces_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file('image.jpg'))[0])
# known_faces_names.append('Name')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = "Unknown"
        
        # If a match was found in known_faces, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_faces_names[first_match_index]
        
        # Draw rectangle around detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw name label below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Facial Recognition Attendance System', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
