import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Sidebar Feature Selection
feature = st.sidebar.selectbox(
    "Choose Feature",
    ["Stress Prediction (Input Data)", "Face Emotion Detection"]
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("model.pkl","rb"))

# -----------------------------
# Feature 1 : Stress Prediction
# -----------------------------
if feature == "Stress Prediction (Input Data)":

    st.title("AI Mental Health Stress Predictor")

    sleep = st.slider("Sleep Hours",1,12)
    study = st.slider("Study Pressure",1,10)
    social = st.slider("Social Interaction",1,10)
    exercise = st.slider("Exercise Hours",0.0,2.0)

    data = np.array([[sleep,study,social,exercise]])

    if st.button("Predict Stress Level"):

        prediction = model.predict(data)

        st.success("Predicted Stress Level: " + str(prediction[0]))

        labels = ["Sleep","Study Pressure","Social","Exercise"]
        values = [sleep,study,social,exercise]

        fig, ax = plt.subplots()

        ax.bar(labels,values)
        ax.set_title("User Lifestyle Factors")

        st.pyplot(fig)

# -----------------------------
# Feature 2 : Face Detection
# -----------------------------
elif feature == "Face Emotion Detection":

    st.title("Real-Time Face Detection (OpenCV)")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    run = st.checkbox("Start Camera")

    camera = cv2.VideoCapture(0)
    frame_window = st.image([])

    while run:

        ret, frame = camera.read()

        if not ret:
            st.write("Camera not working")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(
                frame,
                "Face Detected",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )

        frame_window.image(frame,channels="BGR")
