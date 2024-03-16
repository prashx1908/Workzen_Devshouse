import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

primary_color = "#4B0082"
secondary_color = "#C496A6"
bg_color = "#F4EEFF"
debug_color = "#800080"  # Purple color for debugging info


def set_custom_theme():
    st.markdown(
        f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: 1000px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 5rem;
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

attrition_map = {'No': 0, 'Yes': 1}
business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
education_map = {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
education_field_map = {'Life Sciences': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5, 'Other': 6}
environment_satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
gender_map = {'Male': 1, 'Female': 2}
job_involvement_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
job_role_map = {
    'Sales Executive': 1, 'Research Scientist': 2, 'Laboratory Technician': 3,
    'Manufacturing Director': 4, 'Healthcare Representative': 5, 'Manager': 6,
    'Sales Representative': 7, 'Research Director': 8, 'Human Resources': 9
}
job_satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
marital_status_map = {'Single': 1, 'Married': 2, 'Divorced': 3}
over_time_map = {'No': 0, 'Yes': 1}


age = st.sidebar.number_input('Enter Age', step=1)
attrition = st.sidebar.selectbox('Any plans of resigning?', list(attrition_map.keys()))
business_travel = st.sidebar.selectbox('How comfortable are you with business travel?', list(business_travel_map.keys()))
daily_rate = st.sidebar.number_input('What is your daily rate?', step=1)
distance_from_home = st.sidebar.number_input('Distance From Home', step=1)
education = st.sidebar.selectbox('Maximum education?', list(education_map.keys()))
education_field = st.sidebar.selectbox('Education Field?', list(education_field_map.keys()))
environment_satisfaction = st.sidebar.selectbox('Work environment Satisfaction', list(environment_satisfaction_map.keys()))
gender = st.sidebar.selectbox('Gender', list(gender_map.keys()))
hourly_rate = st.sidebar.number_input('What is your hourly pay?', step=1)
job_involvement = st.sidebar.selectbox('How involved are you in your job?', list(job_involvement_map.keys()))
job_level = st.sidebar.number_input('Job Level?', step=1)
job_role = st.sidebar.selectbox('Job Role?', list(job_role_map.keys()))
job_satisfaction = st.sidebar.selectbox('Job Satisfaction?', list(job_satisfaction_map.keys()))
marital_status = st.sidebar.selectbox('Marital Status?', list(marital_status_map.keys()))
monthly_income = st.sidebar.number_input('Monthly Income?', step=1)
monthly_rate = st.sidebar.number_input('What is your monthly pay?', step=1)
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


input_data = np.array([age, attrition_map[attrition], business_travel_map[business_travel], daily_rate, distance_from_home, education_map[education], 
                       education_field_map[education_field], environment_satisfaction_map[environment_satisfaction], 
                       gender_map[gender], hourly_rate, job_involvement_map[job_involvement], 
                       job_level, job_role_map[job_role], job_satisfaction_map[job_satisfaction], 
                       marital_status_map[marital_status], monthly_income, monthly_rate, num_companies_worked, 
                       over_time_map[over_time], percent_salary_hike, performance_rating, relationship_satisfaction, 
                       stock_option_level, total_working_years, training_times_last_year, years_at_company, 
                       years_in_current_role, years_since_last_promotion, years_with_curr_manager]).reshape(1, -1)

rf_classifier = RandomForestClassifier(random_state=42)
X = df.drop(['WorkLifeBalance'], axis=1)
y = df['WorkLifeBalance']
rf_classifier.fit(X, y)


def predict_outcome(input_data):
    predicted_outcome = rf_classifier.predict(input_data)
    return predicted_outcome


def display_insights(predicted_label):
    if predicted_label == 0:
        st.write("🎉 **Congratulations! You have a healthy work-life balance.** 🎉")
        st.write("- Keep maintaining a balance between your professional and personal life.")
        st.write("- Consider engaging in hobbies or activities outside of work to continue feeling fulfilled.")
    elif predicted_label == 1:
        st.write("⏰ **Your work-life balance seems slightly affected.** ⏰")
        st.write("- Prioritize time management techniques to allocate more time for personal activities.")
        st.write("- Take short breaks during work hours to recharge and avoid burnout.")
        st.write("- Consider discussing workload concerns with your manager for possible adjustments.")
    elif predicted_label == 2:
        st.write("⚖️ **Your work-life balance appears moderately affected.** ⚖️")
        st.write("- Set clear boundaries between work and personal life to maintain separation.")
        st.write("- Schedule regular breaks throughout the day to reduce stress and improve focus.")
        st.write("- Seek support from colleagues or mentors to discuss workload distribution.")
    elif predicted_label == 3:
        st.write("😓 **Your work-life balance seems completely affected.** 😓")
        st.write("- Immediate action is crucial to restore a healthy work-life balance.")
        st.write("- Consider discussing workload concerns with your manager or HR for support.")
        st.write("- Explore options for flexible work arrangements or time-off to recuperate.")


def main():
    logo = 'WORKZEN.png'
    st.image(logo, width=300, caption="Empowering Work-Life Balance")
    
    st.title("WorkZen - Your Integrity Partner")
    st.write("Imagine a life where every developer is empowered with insights into their work-life balance, fostering a culture of well-being and productivity. Our prediction model, built with cutting-edge technology like StreamLit, not only forecasts productivity trends but also instills awareness in users about their work-life balance. By harnessing the power of data validation and settings management, we pave the way for a healthier and more efficient work environment. Join us in revolutionizing workspace productivity management and empowering individuals to thrive both professionally and personally.")

    if st.sidebar.button("Predict Work-Life Balance"):
        prediction = predict_outcome(input_data)
        display_insights(prediction[0])

    st.write("## Debugging Info", className="debug-info")
 
    feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
    st.bar_chart(feature_importances)

    st.write("Input Data Shape:", input_data.shape)
    st.write("Input Data:", input_data)

    predicted_label = predict_outcome(input_data)
    st.write("Predicted Label:", predicted_label)

if __name__ == "__main__":
    set_custom_theme()  
    main()
