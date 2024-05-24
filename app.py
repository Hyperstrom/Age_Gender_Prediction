import cv2
import streamlit as st
import numpy as np
from tensorflow import keras    
from numpy import round
from app2 import*
from io import BytesIO
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(current_dir,'age_gen_net_vgg4.h5')
model = keras.models.load_model(model_path)

def download_image(image_array):
    # Convert the image array back to bytes
    image_bytes = cv2.imencode('.jpg', image_array)[1].tobytes()
    
    # Create a BytesIO object to hold the image bytes
    bytes_io = BytesIO(image_bytes)
    
    # Add a download button
    st.download_button(
        label="Download Image",
        data=bytes_io,
        file_name='image.jpg',
        mime='image/jpeg'
    )
    
def age_gender_predcition(face_resized):
    
    face_resized = face_resized/255.0            
    face_resized = np.expand_dims(face_resized,axis=0)
    pred = model.predict(face_resized)
    age = round(pred[0][0], 2)
    gender = 'female' if pred[1][0] > 0.25 else 'male'
    return age, gender
# Load the pre-trained face detector:

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
st.set_page_config(layout="wide")

def main():
    st.title("Age and gender prediction")

    # Create a column layout with 2/3 of the width for the camera display
    col1, col2 = st.columns([1, 1])
    # person_info = []  # List to store person information
    with col1:
        
        st.header("Camera")
        # Create a VideoCapture object to capture frames from the camera
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        
        start_button_pressed = st.button("start", type="primary")
        stop_button_pressed = st.button("stop", type="primary")
        # st.markdown(
        #     """
        #     <style>
        #     .blue-container {
        #         padding: 20px;
        #         background-color: #cce6ff;
        #         border-radius: 10px;
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True
        # )
        
        while cap.isOpened() and start_button_pressed and not stop_button_pressed :
            
            ret, frame = cap.read()  # Read a frame from the camera
            
            frame =cv2.flip(frame,1)
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
            
            # Iterate over each detected face
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (46, 204, 113), 2)

                # Extract the face region
                face_roi = frame[y:y+h, x:x+w]

                # Resize the face region to (200, 200)
                face_resized = cv2.resize(face_roi, (200, 200))
                # Predict age and gender
                age, gender = age_gender_predcition(face_resized)
                # Display age and gender on the frame
                cv2.putText(frame, f'Age: {age}, Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113 ), 2)
                
                del age
                del gender
            if not ret: 
                
                st.write("The video is stoped")
                break
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")
            
            if stop_button_pressed and not start_button_pressed:
                break
            
        cap.release()
        cv2.destroyAllWindows()
        
    # Display number of persons detected and gender and age details in col2 (sidebar)
    with col2:
        person_info = []
        st.header("upload images to detect gender and age")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
        
        if uploaded_file is not None:
            # Convert the uploaded file to a numpy array
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            
            # Read the image using OpenCV
            image = cv2.imdecode(file_bytes, 1)
            output_image, person_info = detect_faces_and_predict_age_gender(image)
            
            # Display the image
            st.image(output_image, channels="BGR", caption="Uploaded Image")
            for person in person_info:
                st.write("person : ",str(person[0]),"|| Gender : ",str(person[1]),"|| Age : ",str(person[2]))
            
            person_info.clear()
            
            download_image(output_image)
if __name__ == "__main__":
    main()

