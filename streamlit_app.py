import streamlit as st
import cv2
import numpy as np

# Initialize the Streamlit app
st.title('Video Face Recognition App')

# Set up the video stream display
video_stream = st.empty()

# Set up the input form for adding new faces
add_face_form = st.form()
name = add_face_form.text_input('Name')
add_face_button = add_face_form.button('Add Face')

# Set up the output display for recognized faces
recognized_faces = st.empty()

# Set up the video capture object
cap = cv2.VideoCapture(0)

# Load the known faces and labels from the data directory
known_faces_dir = 'data/known_faces'
known_faces = []
known_labels = []
for face_file in os.listdir(known_faces_dir):
    face = cv2.imread(os.path.join(known_faces_dir, face_file), 0)
    known_faces.append(face)
    known_labels.append(face_file.split('.')[0])
#cool
# Set up the face recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(known_faces, np.array(known_labels))

# Perform face recognition on the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_stream.image(frame, width=600)
    
    # Check if the "Add Face" button was clicked
    if add_face_button.was_clicked():
        # Save the current frame as a new known face
        face_file = f'data/known_faces/{name}.jpg'
        cv2.imwrite(face_file, frame)
        st.success(f'Added {name} to known faces')
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
    
    # Recognize the faces
    # Recognize the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = model.predict(roi_gray)
        if confidence < 50:
            label_text = known_labels[label]
        else:
            label_text = 'unknown'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
