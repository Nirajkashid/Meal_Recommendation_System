import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page

# Set page config
st.set_page_config(page_title="Visualizations", layout="wide")

# Load CSS
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Dark Mode Toggle (Optional if implemented earlier)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
dark_mode_toggle = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode_toggle

# Validate that necessary session state data exists
required_keys = [
    'content_recs', 'collaborative_recs', 'recommendation_type', 'bmi',
    'weight', 'height', 'age', 'gender'
]

if not all(k in st.session_state for k in required_keys):
    st.warning("No recommendations generated yet. Please go to the Meal Recommender page first.")
    if st.button("Go to Meal Recommender", use_container_width=True):
        switch_page("Meal_Recommender")
else:
    # Extract data from session state
    content_recs = st.session_state.content_recs
    collaborative_recs = st.session_state.collaborative_recs
    recommendation_type = st.session_state.recommendation_type
    bmi = st.session_state.bmi
    weight = st.session_state.weight
    height = st.session_state.height
    age = st.session_state.age
    gender = st.session_state.gender

    # Sidebar User Info
    st.sidebar.markdown("## User Info")
    st.sidebar.markdown(f"**Gender:** {gender}")
    st.sidebar.markdown(f"**Age:** {age}")
    st.sidebar.markdown(f"**Weight:** {weight} kg")
    st.sidebar.markdown(f"**Height:** {height} cm")
    st.sidebar.markdown(f"**BMI:** {bmi:.1f}")

    st.title("Meal Recommendation Visualizations")
    st.subheader(recommendation_type)

    # --- Content-based Recommendations Visualization ---
    st.markdown("### Content-Based Recommendations")
    if not content_recs.empty:
        fig = px.bar(
            content_recs,
            x='item',
            y='calories',
            color='Ratings',
            title='Calories in Recommended Meals',
            labels={'calories': 'Calories', 'item': 'Meal Item'}
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            content_recs,
            x='protien',
            y='totalfat',
            size='calories',
            color='Ratings',
            hover_name='item',
            title='Protein vs Fat in Recommended Meals'
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No content-based recommendations available.")

    # --- Collaborative Recommendations Visualization ---
    st.markdown("### Collaborative Filtering Recommendations")
    if not collaborative_recs.empty:
        fig3 = px.bar(
            collaborative_recs,
            x='item',
            y='Ratings',
            color='calories',
            title='Ratings of Collaborative Recommendations',
            labels={'Ratings': 'Rating', 'item': 'Meal Item'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No collaborative filtering recommendations available.")

    # Back Navigation Button
    if st.button("← Back to Recommender", use_container_width=True):
        switch_page("Meal_Recommender")

    # Optional Reset Session Button
    if st.sidebar.button("Reset Session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        switch_page("HomePage")
