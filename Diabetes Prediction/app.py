import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Reading the data using pandas
data = pd.read_csv("diabetes-dataset.csv")

# Copy the data into df
df = data.copy(deep=True)

# Replacing zero(0) with NaN
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# Filling all NaN with mean values
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace=True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace=True)
df['BMI'].fillna(df['BMI'].mean(), inplace=True)

# Split independent(x) and dependent(y) data
x = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]]
y = df["Outcome"]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Training the data
KNC = KNeighborsClassifier(n_neighbors=1)
KNC.fit(x_train, y_train)


def prediction(g, b, s, i, bmi):
    y_pred_KNC = KNC.predict([[g, b, s, i, bmi]])
    return y_pred_KNC


st.title("Diabetes Prediction")

name = st.text_input("Patient's Name")
glucose = st.text_input("Glucose (mg/dL) ----- Range: 70-180 mg/dL")
blood_pressure = st.text_input("Blood Pressure (mm Hg) ----- Range: 10-140 mm Hg")
skin_thickness = st.text_input("Skin Thickness (mm) ----- Range: 25-50 mm")
insulin = st.text_input("Insulin (mu U/ml) ----- Range: 15-276 mu U/ml")
bmi = st.text_input("Body Mass Index ----- Range: 10-50")

if st.button("Predict"):
    if not (name and glucose and blood_pressure and skin_thickness and insulin and bmi):
        st.error("Please enter all the details")
    else:
        try:
            glucose = float(glucose)
            blood_pressure = float(blood_pressure)
            skin_thickness = float(skin_thickness)
            insulin = float(insulin)
            bmi = float(bmi)
            p = prediction(glucose, blood_pressure, skin_thickness, insulin, bmi)
            if p[0] == 1:
                st.success("Diabetes: Positive")
            else:
                st.success("Diabetes: Negative")
            st.write(f"Patient's name: {name}")
            st.write(f"Glucose: {glucose}")
            st.write(f"Blood Pressure: {blood_pressure}")
            st.write(f"Skin Thickness: {skin_thickness}")
            st.write(f"Insulin: {insulin}")
            st.write(f"Body Mass Index: {bmi}")
        except ValueError:
            st.error("Please enter valid numeric values")
        

if st.button("Clear"):
    st.text_input("Patient's Name", "")
    st.text_input("Glucose (mg/dL) ----- Range: 70-180 mg/dL")
    st.text_input("Blood Pressure (mm Hg) ----- Range: 10-140 mm Hg")
    st.text_input("Skin Thickness (mm) ----- Range: 25-50 mm")
    st.text_input("Insulin (mu U/ml) ----- Range: 15-276 mu U/ml")
    st.text_input("Body Mass Index ----- Range: 10-50")
