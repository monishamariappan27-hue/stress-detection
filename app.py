import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl","rb"))

st.title("AI Mental Health Stress Predictor")

sleep = st.slider("Sleep Hours",1,12)
study = st.slider("Study Pressure",1,10)
social = st.slider("Social Interaction",1,10)
exercise = st.slider("Exercise Hours",0.0,2.0)

data = np.array([[sleep,study,social,exercise]])

if st.button("Predict Stress Level"):

    prediction = model.predict(data)

    st.success("Predicted Stress Level: " + prediction[0])

    labels = ["Sleep","Study Pressure","Social","Exercise"]
    values = [sleep,study,social,exercise]

    fig, ax = plt.subplots()
    ax.bar(labels,values)

    st.pyplot(fig)