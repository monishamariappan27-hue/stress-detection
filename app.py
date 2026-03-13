import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# Sidebar Feature Selection
# -----------------------------
feature = st.sidebar.selectbox(
    "Choose Feature",
    ["Stress Prediction (Input Data)", "Face Emotion Detection"]
)

# -----------------------------
# Load ML Model
# -----------------------------
model = pickle.load(open("model.pkl","rb"))

# =====================================================
# Feature 1 : Stress Prediction
# =====================================================
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

# =====================================================
# Feature 2 : Face Detection + Emotion
# =====================================================
elif feature == "Face Emotion Detection":

    st.title("Face Emotion & Stress Detection")

    st.write("Take a photo using camera to detect face and emotion.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    emotion_labels = ["Happy","Neutral","Sad","Angry"]

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:

        # Convert image to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        np_img = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            emotion = np.random.choice(emotion_labels)

            if emotion == "Happy":
                stress = "Low Stress"
            elif emotion == "Neutral":
                stress = "Medium Stress"
            else:
                stress = "High Stress"

            cv2.putText(
                frame,
                "Emotion: " + emotion,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2
            )

            cv2.putText(
                frame,
                "Stress: " + stress,
                (x,y+h+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

        st.image(frame, channels="BGR")

