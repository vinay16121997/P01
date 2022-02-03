import cv2
import streamlit as st
from keras.models import load_model

haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_note(img):
    coods = haar.detectMultiScale(img)
    return coods

categories = ["10_note","20_note","50_note","100_note","200_note","500_note","2000_note"]
model = load_model('keras_model.h5')

def classify(img):
  # img = cv2.imread(img)
  img = cv2.resize(img,(224,224))
  y_pred = model.predict(img.reshape(1,224,224,3))
  y_pred = y_pred.argmax()
  output = categories[y_pred]
  return output

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    faces = detect_note(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 3)
        cv2.imshow('Face Detection', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame,(224,224))
    x = classify(img)
    cv2.putText(frame, f'{x}', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

# while run:
#     _, frame = camera.read()
#     faces = detect_note(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 3)
#         cv2.imshow('Face Detection', frame)
#     # FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')