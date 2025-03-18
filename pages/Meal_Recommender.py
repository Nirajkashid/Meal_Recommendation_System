import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from provided CSV content
CSV_CONTENT = '''[paste the entire CSV content here]'''

# Data Loading and Preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv(pd.compat.StringIO(CSV_CONTENT), on_bad_lines='skip', delimiter=',')
    
    # Data preprocessing
    df['calories'] = df['calories'].astype(float)
    df['protien'] = df['protien'].astype(float)
    df['totalfat'] = df['totalfat'].astype(float)
    df['carbs'] = df['carbs'].astype(float)
    df['category'] = df['menu'].apply(lambda x: 'Healthy' if x in ['regular', 'breakfast'] else 'Treat')
    
    return df

df = load_data()

# BMI Calculator
def calculate_bmi(weight, height):
    return weight / ((height/100) ** 2)

# Recommendation Engine
def get_recommendations(bmi, df):
    # Filter based on BMI category
    if bmi < 18.5:
        filtered = df[df['calories'] > 400].sort_values('calories', ascending=False)
        st.session_state.recommendation_type = "High-Calorie Recommendations for Weight Gain"
    elif 18.5 <= bmi < 25:
        filtered = df.sort_values('Ratings', ascending=False)
        st.session_state.recommendation_type = "Balanced Meal Recommendations"
    else:
        filtered = df[df['calories'] < 400].sort_values('calories')
        st.session_state.recommendation_type = "Low-Calorie Recommendations for Weight Management"
    
    # Create feature matrix
    features = filtered[['calories', 'protien', 'totalfat', 'carbs']].values
    indices = filtered.index.values
    
    # Calculate similarity
    if bmi < 18.5:
        weights = [0.1, 0.4, 0.3, 0.2]  # Higher protein focus
    else:
        weights = [0.4, 0.3, 0.2, 0.1]  # Lower calorie focus
    
    similarities = cosine_similarity([weights], features)
    top_indices = similarities.argsort()[0][-10:][::-1]
    
    return filtered.iloc[top_indices]

# Visualization Functions
def plot_calorie_distribution(df):
    fig = px.histogram(df, x='calories', 
                      title='Calorie Distribution of Meals',
                      color='category',
                      nbins=20)
    st.plotly_chart(fig, use_container_width=True)

def plot_nutrition_pie(recommendations):
    fig = px.pie(recommendations, 
                values=['protien', 'totalfat', 'carbs'], 
                names=['Protein', 'Fat', 'Carbs'],
                title='Nutritional Composition')
    st.plotly_chart(fig, use_container_width=True)

# Main Page Layout
st.title("Personalized Meal Recommendations")

# BMI Input Section
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    with col2:
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

    bmi = calculate_bmi(weight, height)
    st.subheader(f"Your BMI: {bmi:.1f}")
    
    if bmi < 18.5:
        st.warning("Underweight - Recommending high-calorie meals")
    elif 18.5 <= bmi < 25:
        st.success("Normal Weight - Recommending balanced meals")
    else:
        st.error("Overweight/Obesity - Recommending low-calorie meals")

# Recommendations and Visualizations
if st.button("Generate Recommendations"):
    recommendations = get_recommendations(bmi, df)
    
    st.header(st.session_state.recommendation_type)
    
    # Display recommendations
    cols = st.columns(3)
    for idx, (_, row) in enumerate(recommendations.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="meal-card">
                <h4>{row['item']}</h4>
                <p>Calories: {row['calories']:.0f}</p>
                <p>Protein: {row['protien']}g</p>
                <p>Carbs: {row['carbs']}g</p>
                <p>Fat: {row['totalfat']}g</p>
                <p>Rating: {row['Ratings']}/20</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show visualizations
    st.header("Nutrition Insights")
    plot_calorie_distribution(df)
    plot_nutrition_pie(recommendations)
