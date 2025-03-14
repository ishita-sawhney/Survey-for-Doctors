import os
import pandas as pd
import joblib
import datetime
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")  
    return df

def preprocess_data(df):
    df['Login Time'] = pd.to_datetime(df['Login Time'], errors='coerce').dt.hour * 60 + pd.to_datetime(df['Login Time'], errors='coerce').dt.minute
    df['Logout Time'] = pd.to_datetime(df['Logout Time'], errors='coerce').dt.hour * 60 + pd.to_datetime(df['Logout Time'], errors='coerce').dt.minute
    df['Usage Time (mins)'] = df['Usage Time (mins)'].fillna(df['Usage Time (mins)'].median())
    df['Count of Survey Attempts'] = df['Count of Survey Attempts'].fillna(0)
    df['Survey Frequency'] = df['Count of Survey Attempts'] / (df['Usage Time (mins)'] + 1)  
    label_encoders = {}
    for col in ['Speciality', 'Region', 'State']:
        le = LabelEncoder()
        df[col + " Encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        joblib.dump(le, f'{col}_encoder.pkl')

    joblib.dump(label_encoders, 'label_encoders.pkl')  
    return df


def train_model(df):
    feature_columns = ['Speciality Encoded', 'Region Encoded', 'State Encoded', 'Login Time', 'Logout Time', 'Usage Time (mins)', 'Survey Frequency']
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
    buffer_time = 5 
    input_data = input_data[
        ((input_data['Login Time'] <= time_input) & (input_data['Logout Time'] >= time_input)) |  
        ((input_data['Login Time'] >= time_input - buffer_time) & (input_data['Login Time'] <= time_input + buffer_time))]
    if input_data.empty:
        return pd.DataFrame(columns=['S.No', 'NPI', 'Speciality', 'Region', 'State'])

    feature_columns = joblib.load('feature_columns.pkl')
    predictions = model.predict(input_data[feature_columns])
    selected_doctors = input_data[predictions == 1][['NPI', 'Speciality', 'Region', 'State']]
    selected_doctors.insert(0, 'S.No', range(1, len(selected_doctors) + 1))  
    return selected_doctors


def main():
    st.set_page_config(page_title="Doctor Survey Prediction", layout="centered")
    st.title("Doctor Survey Prediction System")
    st.subheader("Predict which doctors are likely to fill out surveys based on login times.")
    st.write("Select a time below to see which doctors are more likely to fill out surveys during that period.")
    time_input = st.slider(
        "Choose a time",
        min_value=datetime.time(0, 0),
        max_value=datetime.time(23, 59),
        value=datetime.time(12, 0),
        step=datetime.timedelta(minutes=1),
        format="HH:mm"
    )
    time_input_minutes = time_input.hour * 60 + time_input.minute
    st.write("---")
    if st.button("Predict"):
        with st.spinner("Processing..."):
            df = load_data("dummy_npi_data.xlsx")
            df = preprocess_data(df)
            model = joblib.load('doctor_survey_model.pkl')
            result = predict_doctors(time_input_minutes, model, df)
            if not result.empty:
                st.success(f"✅ {len(result)} doctors found!")
                st.dataframe(result.set_index("S.No"), width=800, height=500)
                st.download_button("Download CSV", data=result.to_csv(index=False), file_name="predicted_doctors.csv", mime="text/csv")
            else:
                st.warning("⚠ No doctors found at this time.")

if __name__ == "__main__":
    main()
