#language vs placementcorellation
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

st.title("Language vs Placement Correlation Analysis")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df=pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())
    #data preprocessing
    df["Known_Languages"]= df["Known_Languages"].apply(lambda x: x.split(", "))
    mlb=MultiLabelBinarizer()
    language_df=pd.DataFrame(mlb.fit_transform(df["Known_Languages"]), columns=mlb.classes_)
    df=pd.concat([df, language_df], axis=1)
    #now we create a model
    X=df[language_df.columns]
    y=df['Package_LPA']
    model=DecisionTreeRegressor()
    model.fit(X, y)
    #prediction
    st.subheader("Predict salary based on known languages")
    #multi language selection
    selected_langs=st.multiselect("Select Known Languages", options=X.columns)
    button=st.button("Predict Package")
    if button:
        input_data=[1 if lang in selected_langs else 0 for lang in X.columns]
        predicted_package=model.predict([input_data])[0]
        st.success(f"The predicted package is: {predicted_package:.2f} LPA")
    