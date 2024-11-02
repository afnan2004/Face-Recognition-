import cv2
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as K

# Clear any existing models/sessions
K.clear_session()

# Define the model architecture first
model = Sequential([
    Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    Conv2D(512, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

# Now load the weights
model.load_weights("facialemotionmodel.h5")

# Rest of your code
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    if not i:
        print("Error capturing video")
        break
        
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p,q), (p+r,q+s), (255,0,0), 2)
            image = cv2.resize(image, (48,48))
            img = extract_features(image)
            pred = model.predict(img, verbose=0)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, f'{prediction_label}', (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
        
        cv2.imshow("Output", im)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to quit
            break
            
    except cv2.error as e:
        print(f"CV2 Error: {e}")
        continue
    except Exception as e:
        print(f"Other Error: {e}")
        continue

webcam.release()
cv2.destroyAllWindows()