import cv2
import numpy as np
from tensorflow.keras.models import load_model
from numpy import round
import os

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
person_info = []
current_dir = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(current_dir,'age_gen_net_vgg4.h5')
model = load_model(model_path)
def age_gender_predcition(face_resized):
    
    face_resized = face_resized/255.0            
    face_resized = np.expand_dims(face_resized,axis=0)
    pred = model.predict(face_resized)
    age = round(pred[0][0], 2)
    gender = 'female' if pred[1][0] > 0.35 else 'male'
    return age, gender ,pred[1][0]

def detect_faces_and_predict_age_gender(image):
    height, width , _ = image.shape

    aspect_ratio = width / height
    new_height = int(1080/aspect_ratio)

    image = cv2.resize(image,(1080, new_height))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        cv2.rectangle(image, (x, y), (x+w, y+h), (46, 204, 113), 2)

        # Extract the face region
        face_roi = image[y:y+h, x:x+w]

        # Resize the face region to (200, 200)
        face_resized = cv2.resize(face_roi, (200, 200))
        # Predict age and gender
        age, gender, pred = age_gender_predcition(face_resized)
        print(pred)
        # Display age and gender on the frame Gender: {gender}
        cv2.putText(image, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113 ), 2)
        cv2.putText(image, f'Gender: {gender}', (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113 ), 2)
        person_info.append((f"{len(person_info)+1}  ", f"{gender}", f"{age}"))
        del age
        del gender
        del pred
        del face_resized
    return image, person_info 

cv2.destroyAllWindows()
