import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")  
    return df

def preprocess_data(df):
    df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce').dt.hour
    df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce').dt.hour
    df['Usage Time (mins)'] = df['Usage Time (mins)'].fillna(df['Usage Time (mins)'].median())
    df['Count of Survey Attempts'] = df['Count of Survey Attempts'].fillna(0)

    label_encoders = {}
    for col in ['Speciality', 'Region', 'State']:
        le = LabelEncoder()
        df[col + " Encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        joblib.dump(le, f'{col}_encoder.pkl')

    joblib.dump(label_encoders, 'label_encoders.pkl')  
    return df

def train_model(df):
    feature_columns = ['Speciality Encoded', 'Region Encoded', 'State Encoded', 'Login Time', 'Logout Time', 'Usage Time (mins)', 'Count of Survey Attempts']
    X = df[feature_columns]
    y = (df['Count of Survey Attempts'] > 1).astype(int)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'doctor_survey_model.pkl') 
    joblib.dump(feature_columns, 'feature_columns.pkl')
    return model

def predict_doctors(time_input, model, df):
    input_data = df.copy()
    input_data = input_data[(input_data['Login Time'] > time_input) | (input_data['Logout Time'] < time_input)]
    feature_columns = joblib.load('feature_columns.pkl')
    predictions = model.predict(input_data[feature_columns])
    selected_doctors = input_data[predictions == 1][['NPI', 'Speciality', 'Region', 'State']]
    selected_doctors = selected_doctors.reset_index(drop=True)  
    selected_doctors.insert(0, 'S.No', range(1, len(selected_doctors) + 1))  
    return selected_doctors.reset_index(drop=True)

def main():
    st.set_page_config(page_title="Doctor Survey Prediction", layout="centered")
    st.title("Doctor Survey Prediction System")
    st.subheader("Predict which doctors are likely to fill out surveys based on login times.")
    
    st.write("Enter a time below to see which doctors are more likely to fill out surveys during that period.")
    
    time_input = st.slider("Enter Time (0-23)", min_value=0, max_value=23, step=1, value=12, label_visibility="collapsed")
    
    st.write("---")  # A line separator for better UI clarity

    if st.button("Predict"):
        with st.spinner("Processing..."):
            df = load_data("dummy_npi_data.xlsx")
            df = preprocess_data(df)
            if not os.path.exists('doctor_survey_model.pkl') or not os.path.exists('feature_columns.pkl'):
                model = train_model(df) 
            else:
                model = joblib.load('doctor_survey_model.pkl')
            result = predict_doctors(time_input, model, df)
            st.success("Prediction Complete!")
            st.dataframe(result.set_index("S.No"), width=800, height=500)
            result.to_csv("predicted_doctors.csv", index=False)
            st.download_button("Download CSV", data=result.to_csv(index=False), file_name="predicted_doctors.csv", mime="text/csv")

if __name__ == "__main__":
    main()
