import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("preprocess.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('ML Heart Attack Prediction Model Deployment')
  
    age = st.slider('input Age',min_value=0.0,max_value=100.0,value=1.0)
    sexopt = st.selectbox("Sex", ["Male", "Female"])
    sex_mapping = {
    "Male": 1,
    "Female": 0
    }
    sex = sex_mapping[sexopt]
    cp = st.slider('input CP',min_value=0.0,max_value=100.0,value=1.0)
    trestbps = st.number_input('input Trestbps', min_value=0.0, max_value=1000.0, value=1.0)
    chol = st.number_input('input Kolestrol', min_value=0.0, max_value=1000.0, value=1.0)
    fbs = st.selectbox("fbs", [1, 0])
    restecg = st.selectbox("restecg", [1, 0])
    thalach = st.number_input('input thalach', min_value=0.0, max_value=20.0, value=1.0)
    exang = st.selectbox("exang", [1, 0])
    oldpeak = st.number_input('input oldpeak', min_value=0.0, max_value=20.0, value=0.1)
    slopes = st.number_input('input slope', min_value=0.0, max_value=1000.0, value=1.0)
    ca = st.number_input('input ca', min_value=0.0, max_value=1000.0, value=1.0)
    thal= st.number_input('input thal', min_value=0.0, max_value=1000.0, value=1.0)

    if st.button('Make Prediction'):
        features = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slopes,ca,thal]
        result = pred(features)
        st.success(f'The prediction is: {result}')

def pred(features):
    input_array = np.array(features).reshape(1, -1)
    xscaled = scaler.transform(input_array)
    prediction = model.predict(xscaled)
    return prediction[0]

if __name__ == '__main__':
    main()


