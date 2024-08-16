import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

primary_color = "#4B0082"
secondary_color = "#C496A6"
bg_color = "#F4EEFF"
debug_color = "#800080"

def set_custom_theme():
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: 1000px;
            padding: 2rem;
            font-weight: bold;
        }}
        .reportview-container .main{{
            color: #6A0572;
            background-color: {bg_color};
        }}
        .sidebar .sidebar-content{{
            background-color: {primary_color};
            color: white;
        }}
        .Widget>label{{
            color: {primary_color};
            font-weight: bold;
        }}
        .stTextInput>div>div>div>input,
        .stNumberInput>div>div>div>input,
        .stTextArea>div>div>textarea,
        .stDateInput>div>div>div>input,
        .stSelectbox>div>div>div>div{{
            background-color: white;
            color: #6A0572;
            border-color: {secondary_color};
            border-width: 2px;
            border-radius: 5px;
            padding: 8px;
        }}
        div.stButton>button{{
            color: white;
            background-color: {secondary_color};
            border-radius: 5px;
            padding: 10px 15px;
            font-weight: bold;
        }}
        .st-bw{{
            color: white !important;
            background-color: {primary_color} !important;
            font-weight: bold;
        }}
        .debug-info {{
            color: {debug_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

df = pd.read_csv('extended_file6.csv')
df = df.drop(['Department'], axis=1)

st.sidebar.title("Hey Developers! Kindly Enter the Details")

# Mapping categorical inputs to numerical values
attrition_map = {'No': 0, 'Yes': 1}
business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
education_map = {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
education_field_map = {'Computer Science': 1, 'Medical': 2, 'Engineering': 3, 'Technical Degree': 4, 'Human Resources': 5, 'Other': 6}
environment_satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
gender_map = {'Male': 0, 'Female': 1}
job_involvement_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
job_role_map = {
    'Software Developer': 0, 'Data Scientist': 1, 'Security Engineer': 2,
    'Research Engineer': 3, 'ML Engineer': 4, 'Manager': 5,
    'Cloud Architect': 6, 'Web Developer': 7, 'Network Engineer': 8
}
job_satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
over_time_map = {'No': 0, 'Yes': 1}

# Sidebar inputs
age = st.sidebar.number_input('Enter Age', step=1)
attrition = st.sidebar.selectbox('Any plans of resigning?', list(attrition_map.keys()))
business_travel = st.sidebar.selectbox('How comfortable are you with business travel?', list(business_travel_map.keys()))
daily_rate = st.sidebar.number_input('What is your daily rate? in $USD', step=1)
distance_from_home = st.sidebar.number_input('Distance From Home in Miles', step=1)
education = st.sidebar.selectbox('Maximum education?', list(education_map.keys()))
education_field = st.sidebar.selectbox('Education Field?', list(education_field_map.keys()))
environment_satisfaction = st.sidebar.selectbox('Work environment Satisfaction', list(environment_satisfaction_map.keys()))
gender = st.sidebar.selectbox('Gender', list(gender_map.keys()))
hourly_rate = st.sidebar.number_input('What is your hourly pay? in $USD', step=1)
job_involvement = st.sidebar.selectbox('How involved are you in your job?', list(job_involvement_map.keys()))
job_level = st.sidebar.number_input('Job Level?', step=1)
job_role = st.sidebar.selectbox('Job Role?', list(job_role_map.keys()))
job_satisfaction = st.sidebar.selectbox('Job Satisfaction?', list(job_satisfaction_map.keys()))
marital_status = st.sidebar.selectbox('Marital Status?', list(marital_status_map.keys()))
monthly_income = st.sidebar.number_input('Monthly Income? in $USD', step=1)
monthly_rate = st.sidebar.number_input('What is your monthly pay? in $USD', step=1)
num_companies_worked = st.sidebar.number_input('Number of previous company experience?', step=1)
over_time = st.sidebar.selectbox('Do you find yourself doing overtime?', list(over_time_map.keys()))
percent_salary_hike = st.sidebar.number_input('Percentage of salary hike?', step=1)
performance_rating = st.sidebar.selectbox('Performance Rating?', [1, 2, 3, 4, 5])
relationship_satisfaction = st.sidebar.selectbox('Relationship with current manager?', [1, 2, 3, 4])
stock_option_level = st.sidebar.selectbox('Stock Option Level?', [0, 1])
total_working_years = st.sidebar.number_input('Total Working Years?', step=1)
training_times_last_year = st.sidebar.number_input('Training Times Last Year?', step=1)
years_at_company = st.sidebar.number_input('Years at Company?', step=1)
years_in_current_role = st.sidebar.number_input('Years in Current Role?', step=1)
years_since_last_promotion = st.sidebar.number_input('Years Since Last Promotion?', step=1)
years_with_curr_manager = st.sidebar.number_input('Years with Current Manager?', step=1)

# Prepare input data for prediction
input_data = np.array([age, attrition_map[attrition], business_travel_map[business_travel], daily_rate, distance_from_home, 
                       education_map[education], education_field_map[education_field], 
                       environment_satisfaction_map[environment_satisfaction], gender_map[gender], 
                       hourly_rate, job_involvement_map[job_involvement], job_level, 
                       job_role_map[job_role], job_satisfaction_map[job_satisfaction], 
                       marital_status_map[marital_status], monthly_income, monthly_rate, 
                       num_companies_worked, over_time_map[over_time], percent_salary_hike, 
                       performance_rating, relationship_satisfaction, stock_option_level, 
                       total_working_years, training_times_last_year, years_at_company, 
                       years_in_current_role, years_since_last_promotion, 
                       years_with_curr_manager]).reshape(1, -1)

# Train model with a train/test split for better validation
X = df.drop(['WorkLifeBalance'], axis=1)
y = df['WorkLifeBalance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

def predict_outcome(input_data):
    predicted_outcome = rf_classifier.predict(input_data)
    return predicted_outcome

def display_insights(predicted_label):
    if predicted_label == 1:
        st.write("🎉 **Congratulations! You have a healthy work-life balance.** 🎉")
    elif predicted_label == 2:
        st.write("⏰ **Your work-life balance seems slightly affected.** ⏰")
    elif predicted_label == 3:
        st.write("⚖️ **Your work-life balance appears moderately affected.** ⚖️")
    elif predicted_label == 4:
        st.write("😓 **Your work-life balance seems completely affected.** 😓")

def main():
    st.image('WORKZEN.png', width=300, caption="Empowering Work-Life Balance")
    st.title("WorkZen - Your Integrity Partner")

    set_custom_theme()

    if st.sidebar.button("Submit"):
        predicted_outcome = predict_outcome(input_data)
        st.success(f"Predicted Work-Life Balance: {predicted_outcome[0]}")
        display_insights(predicted_outcome[0])

if __name__ == "__main__":
    main()
