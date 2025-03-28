import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.switch_page_button import switch_page
from streamlit.runtime.scriptrunner import get_pages


# Set page config
st.set_page_config(
    page_title="Meal Recommendation System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #ff9966, #ff5e62);
            color: #ffffff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }
        .title-text {
            font-size: 3.5em;
            text-shadow: 2px 2px 4px #000000;
        }
    </style>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 3])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046784.png", width=150)

with col2:
    st.markdown('<p class="title-text">Meal Recommendation System</p>', unsafe_allow_html=True)
    st.markdown("**Discover personalized meal recommendations based on your preferences!**")

st.markdown("---")

# Features section
st.header("Features")
features = [
    ("🍳", "Personalized Recommendations", "Get meal suggestions based on your dietary needs"),
    ("📊", "Nutrition Insights", "Detailed nutritional breakdown for each meal"),
    ("❤️", "Health First", "Recipes tailored to support your health goals"),
    ("⭐", "User Favorites", "Explore our most popular recipes"),
]

cols = st.columns(4)
for i, (icon, title, text) in enumerate(features):
    with cols[i]:
        st.markdown(f"<h3>{icon} {title}</h3>", unsafe_allow_html=True)
        st.markdown(f"<small>{text}</small>", unsafe_allow_html=True)

# Action buttons
st.markdown("---")
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🍔 Get Recommendations", use_container_width=True):
        st.switch_page("pages/Meal_Recommender.py")

with col_btn2:
    if st.button("📊 View Visualizations", use_container_width=True):
        st.switch_page("pages/Visualizations.py")

