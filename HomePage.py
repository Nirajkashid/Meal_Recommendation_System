import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Meal Recommender", layout="wide")

# Load CSS
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Load Lottie animation
def load_lottie(path: str):
    with open(path, "r") as p:
        return json.load(p)
lottie_path = load_lottie("./ani.json")

# Home Page Layout
col1, col2 = st.columns([2, 3])
with col1:
    st.title("Smart Meal Recommender")
    st.markdown("""
    <div style='border-left: 5px solid #FF4B4B; padding-left: 1rem;'>
    <h3 style='color: #FF4B4B;'>Get Personalized Meal Recommendations Based on Your BMI</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Get Started →", use_container_width=True):
        switch_page("Meal_Recommender")

with col2:
    st_lottie(lottie_path, height=400, key="home_animation")