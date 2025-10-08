#Student performance analysis

import streamlit as st
import pandas as pd
import joblib   

#load model
model=joblib.load("student_performance_model.pkl")
technology_encoder=joblib.load("technology_encoder.pkl")
grade_encoder=joblib.load("grade_encoder.pkl")

st.subheader("Student Performance Analysis and Prediction App")
st.write("Fill the student scores below to predict their final grade")

#user input
technology=st.selectbox("Technology",technology_encoder.classes_)
welcome_test=st.slider("Welcome Test",30,50,40)
presentation=st.slider("Presentation",90,150,120)
mini_projects=st.slider("Mini Project",60,100,80)
hrskills=st.slider("HR Skills",90,150,120)
project_presentation=st.slider("Project Presentation",160,250,205)
project_submission=st.slider("Project Submission",70,100,85)
attendance=st.slider("Attendence",70,100,85)
discipline=st.slider("Discipline",60,100,80)

#now take a button
button= st.button("Predict ")
if button:
    #preprocess input
    input_data=pd.DataFrame({
        "Technology":[technology_encoder.transform([technology])[0]],
        "Welcome_test":[welcome_test],
        "Presentation":[presentation],
        "Mini_projects":[mini_projects],
        "Hrskills":[hrskills],
        "Project_presentation":[project_presentation],
        "Project_submission":[project_submission],
        "Attendance":[attendance],
        "Discipline":[discipline]
        })
    prediction=model.predict(input_data)
    result=grade_encoder.inverse_transform(prediction)[0]
    st.success(f"The predicted final grade is: {result}**")
