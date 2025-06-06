import json
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr


# Set page config
st.set_page_config(page_title="Meal Recommender", layout="wide")

# Load CSS
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# Optional: Reset session button
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session reset! Starting fresh...")

# Load Lottie animation
def load_lottie(path: str):
    with open(path, "r") as p:
        return json.load(p)

lottie_path = load_lottie("./ani.json")

# Home Page Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.title("Meal Recommendation System")

    st.markdown("""
    <div style='border-left: 5px solid #FF4B4B; padding-left: 1rem;'>
        <h3 style='color: #FF4B4B;'>Get Personalized Meal Recommendations Based on Your BMI</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
        <h4>Features:</h4>
        <ul>
            <li>BMI-based meal recommendations</li>
            <li>Content-based filtering for nutrition matching</li>
            <li>Collaborative filtering with similar user preferences</li>
            <li>Detailed nutrition insights</li>
            <li>Interactive data visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



with col2:
    st_lottie(lottie_path, height=400, key="home_animation")   
