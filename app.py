import streamlit as st
import joblib
import pandas as pd
import os
import datetime

st.set_page_config(page_title="Indian Property Predictor", page_icon="🇮🇳", layout="centered")

# Ensure all ML files exist
required_files = ['ghormach_model.pkl', 'le_country.pkl', 'le_state.pkl', 'le_city.pkl', 'le_type.pkl', 'le_condition.pkl']
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Missing file: {file}. Please run `python train_model.py` first.")
        st.stop()

# Load models and encoders
model = joblib.load('ghormach_model.pkl')
le_country = joblib.load('le_country.pkl')
le_state = joblib.load('le_state.pkl')
le_city = joblib.load('le_city.pkl')
le_type = joblib.load('le_type.pkl')
le_condition = joblib.load('le_condition.pkl')

# --- BULLETPROOF INDIAN GEOGRAPHY DICTIONARY ---
# This guarantees no outside countries or wrong cities will ever appear.
indian_geography = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Naharlagun"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Ambala"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubli", "Belagavi"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"],
    "Manipur": ["Imphal", "Thoubal"],
    "Meghalaya": ["Shillong", "Tura"],
    "Mizoram": ["Aizawl", "Lunglei"],
    "Nagaland": ["Dimapur", "Kohima"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Chandigarh"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "Sikkim": ["Gangtok", "Namchi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad"],
    "Tripura": ["Agartala", "Dharmanagar"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Noida"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Roorkee", "Rishikesh"],
    "West Bengal": ["Kolkata", "Howrah", "Darjeeling", "Siliguri"],
    "Delhi": ["New Delhi", "Delhi Cantonment"],
    "Jammu and Kashmir": ["Srinagar", "Jammu", "Anantnag"]
}

st.title("🇮🇳 Indian Property AI Predictor")
st.markdown("Estimate property values across Indian states and cities.")
st.divider()

# --- GEOGRAPHIC CASCADING UI (Strictly India) ---
st.subheader("1. Location Details")
geo_col1, geo_col2 = st.columns(2)

with geo_col1:
    # Get just the keys (States) from our dictionary
    states = sorted(list(indian_geography.keys()))
    selected_state = st.selectbox("State", states)

with geo_col2:
    # Get the specific cities for the chosen state from the dictionary
    cities = sorted(indian_geography[selected_state])
    selected_city = st.selectbox("City", cities)

# --- CUSTOM PROPERTY DETAILS UI ---
st.subheader("2. Property Features")
prop_col1, prop_col2, prop_col3 = st.columns(3)

with prop_col1:
    current_year = datetime.date.today().year
    year = st.number_input("Year", min_value=2000, max_value=2050, value=current_year)
    prop_type = st.selectbox("Type", le_type.classes_)

with prop_col2:
    size = st.number_input("Size (sqm)", min_value=10, max_value=1000, value=100, step=10)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6])

with prop_col3:
    condition = st.selectbox("Condition", le_condition.classes_)
    distance = st.slider("Distance (km)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

st.divider()

# --- PREDICTION ENGINE & SAFETY NET ---
if st.button("Calculate Property Value", type="primary", use_container_width=True):
    
    if selected_city not in le_city.classes_:
        st.warning(f"⚠️ **AI Alert:** We don't have historical housing data for **{selected_city}** yet!")
        st.info("💡 **Try selecting a city the AI has studied:** Bengaluru, Mumbai, New Delhi, Chennai, or Hyderabad.")
    
    else:
        try:
            # Predict Logic
            country_enc = le_country.transform(['India'])[0]
            state_enc = le_state.transform([selected_state])[0]
            city_enc = le_city.transform([selected_city])[0]
            type_enc = le_type.transform([prop_type])[0]
            condition_enc = le_condition.transform([condition])[0]
            
            input_data = pd.DataFrame([[year, country_enc, state_enc, city_enc, type_enc, size, bedrooms, condition_enc, distance]], 
                                      columns=['Year', 'Country_Enc', 'State_Enc', 'City_Enc', 'Type_Enc', 'Size_sqm', 'Bedrooms', 'Condition_Enc', 'Distance_km'])
            
            prediction_usd = model.predict(input_data)[0]
            
            exchange_rate = 86.50
            prediction_inr = prediction_usd * exchange_rate
            
            st.subheader(f"Estimated Value in {selected_city}")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="Price in INR", value=f"₹{prediction_inr:,.0f}")
            with res_col2:
                st.metric(label="Price in USD", value=f"${prediction_usd:,.0f}")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")