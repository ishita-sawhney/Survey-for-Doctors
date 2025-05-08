import os
import pandas as pd
import joblib
import datetime
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Doctor Survey Prediction", layout="wide")

st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
            
        .center-table {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80% !important;
        }

        /* Ensure the table takes full width in its container */
        .stDataFrame {
            width: 100% !important;
            max-width: 100% !important;
        }
            
        .subheader {
            font-size: 24px;
            color: #333;
            text-align: center;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            transition: all 0.3s ease;
            transform: scale(1);
            font-size: 16px;
            border-radius: 10px;
            padding: 12px 24px;
            border: none;
        }
            
        .stButton > button:hover {
    color: white !important;
    transform: scale(1.1);
}
            
        .stButton > button:active {
    color: white !important;
}

/* Prevent text color from changing on focus */
.stButton > button:focus {
    color: white !important;
    outline: none;
}
        
        .stSlider>div>div>input {
            border-radius: 8px;
            padding: 8px;
            background-color: #f1f1f1;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 30px;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        .main {
            background-image: url('https://your_image_url_here.com'); 
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

# Load data function
def load_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")  
    return df

# Preprocess data function
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

# Train model function
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

# Prediction function
def predict_doctors(time_input, model, df):
    input_data = df.copy()
    buffer_time = 5 
    input_data = input_data[(
        (input_data['Login Time'] <= time_input) & (input_data['Logout Time'] >= time_input)) |  
        ((input_data['Login Time'] >= time_input - buffer_time) & (input_data['Login Time'] <= time_input + buffer_time))]

    if input_data.empty:
        return pd.DataFrame(columns=['S.No', 'NPI', 'Speciality', 'Region', 'State'])

    feature_columns = joblib.load('feature_columns.pkl')
    predictions = model.predict(input_data[feature_columns])
    selected_doctors = input_data[predictions == 1][['NPI', 'Speciality', 'Region', 'State']]
    selected_doctors.insert(0, 'S.No', range(1, len(selected_doctors) + 1))  
    return selected_doctors

# Main function
def main():
    # Header and Subheader
    st.markdown('<h1 class="title">Doctor Survey Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Predict which doctors are likely to fill out surveys based on login times.</h3>', unsafe_allow_html=True)
    
    # Slider for selecting time
    time_input = st.slider(
        "Choose a time",
        min_value=datetime.time(0, 0),
        max_value=datetime.time(23, 59),
        value=datetime.time(12, 0),
        step=datetime.timedelta(minutes=1),
        format="HH:mm",
        help="Use this slider to select the time to predict doctor survey attempts."
    )
    
    time_input_minutes = time_input.hour * 60 + time_input.minute
    st.write("---")
    
    # Using columns to place the button and table separately
    col1, col2 = st.columns([1, 3])  # 1:3 column distribution
    
    with col1:  # Predict button on the left
        if st.button("Predict", key="predict_button"):
            with st.spinner("🔄 Predicting doctor survey attempts... Please wait."):
                df = load_data("dummy_npi_data.xlsx")
                df = preprocess_data(df)
                model = joblib.load('doctor_survey_model.pkl')
                result = predict_doctors(time_input_minutes, model, df)
                
                # Now we will display the result in the center column
                with col2:  # Display the table in the center column
                    if not result.empty:
                        st.success(f"✅ {len(result)} doctors found!")
                        st.dataframe(result.set_index("S.No"), use_container_width=True, height=500)
                        st.download_button("Download CSV", data=result.to_csv(index=False), file_name="predicted_doctors.csv", mime="text/csv")
                    else:
                        st.warning("⚠ No doctors found at this time.")

# Run the app
if __name__ == "__main__":
    main()
