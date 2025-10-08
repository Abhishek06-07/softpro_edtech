#Course Recommendation system.
# IMPORT ESSENTIAL LIBRARIES.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and preprocessing data.
df=pd.read_excel("MIS Data.xlsx")
# Remove dots (.)or white spaces Replace with Underscore(_) or Highfan(-).
df.columns=df.columns.str.strip().str.replace(" ","").str.replace("-","")
df['College']=df['College'].fillna('Not Provided')
df['Branch']=df['Branch'].str.strip().str.title()
df['Course']=df['Course'].str.strip().str.title()
df['Subject']=df['Subject'].str.strip().str.title()

# Encode features and train  ML mopdel for prediction.
features=['Branch','College','Course','Year']
target='Subject'
df_ml=df.dropna(subset=features+[target])
# For encode.
encoders={}
for col in features+[target]:
    le=LabelEncoder()
    df_ml[col]=le.fit_transform(df_ml[col])
    encoders[col]=le
# seperate features and target.
X=df_ml[features]
y=df_ml[target]
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)

# Take inputs and encodes and also decode the output(target).

def predict_subjects_ml(branch,college,course,year,top_n=3):
    input_df=pd.DataFrame([[branch,college,course,year]],columns=features)
    for col in  features:
        input_df[col]=encoders[col].transform(input_df[col])
    probs=model.predict_proba(input_df)[0]
    top_indices=np.argsort(probs)[::-1][:top_n]
    subject_names=encoders[target].inverse_transform(top_indices)
    return list(zip(subject_names,probs[top_indices]))



# Streamlit UI.
st.title("Course Recommendation System")
st.markdown("Get Top Recommended Course Based On Your Branch, College, Course and Year")

# Sidebar inputs.
branches=sorted(df['Branch'].dropna().unique())
colleges=sorted(df['College'].dropna().unique())
courses=sorted(df['Course'].dropna().unique())
years=sorted(df['Year'].dropna().unique())

# For create dropdown.
selected_branch=st.selectbox("Select Branch",branches)
selected_college=st.selectbox("Select College",colleges)
selected_course=st.selectbox("Select Course",courses)
selected_year=st.selectbox("Select Year",years)

# Create Button for prediction.
button=st.button("Recommend Subject (Ml-Based)")
if button:
    ml_recommendations=predict_subjects_ml(selected_branch,selected_college,selected_course,selected_year)
    st.subheader("ML-Based Recommended Subjects")
    for i, (subject,score) in enumerate(ml_recommendations,1):
        st.markdown(f"{i}.* {subject}* - confidence :{score:.2f}")