import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import io
import random
from streamlit_extras.switch_page_button import switch_page
from fpdf import FPDF

st.set_page_config(page_title="Dynamic Meal Recommender", layout="wide")

# Initialize session keys if missing
for key in ['rec_type', 'content_recs', 'collaborative_recs', 'bmi']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'rec_type' not in st.session_state:
    st.session_state.rec_type = None
if 'content_recs' not in st.session_state:
    st.session_state.content_recs = None
if 'collaborative_recs' not in st.session_state:
    st.session_state.collaborative_recs = None
if 'bmi' not in st.session_state:
    st.session_state.bmi = None
    
# Display Content-Based Recommendations
if st.session_state.content_recs is not None and not st.session_state.content_recs.empty:
    st.header(st.session_state.rec_type)

    cols = st.columns(3)

    for idx, (_, row) in enumerate(st.session_state.content_recs.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="meal-card">
                    <h4>{row['item']}</h4>
                    <p>Calories: {row['calories']}</p>
                    <p>Protein: {row['protien']}</p>
                    <p>Carbs: {row['carbs']}</p>
                </div>
            """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No recommendations to display. Please generate them first!")
    if st.button("Go to Meal Recommender"):
        switch_page("Meal_Recommender")


# Load dataset from provided CSV content
CSV_CONTENT = """item id,item,servesize,calories,protien,totalfat,satfat,transfat,cholestrol,carbs,sugar,addedsugar,sodium,menu,Ratings
0,McVeggie Burger,168 ,402,10.24,13.83,5.34,0.16,2.49,56.54,7.9,4.49,706.13,regular,1
1,McAloo Tikki Burger,146 ,339,8.5,11.31,4.27,0.2,1.47,5.27,7.05,4.07,545.34,regular,15
2,McSpicy Paneer Burger,199 ,652,20.29,39.45,17.12,0.18,21.85,52.33,8.35,5.27,1074.58,regular,19
3,Spicy Paneer Wrap,250 ,674,20.96,39.1,19.73,0.26,40.93,59.27,3.5,1.08,1087.46,regular,10
4,American Veg Burger,177 ,512,15.3,23.45,10.51,0.17,25.24,56.96,7.85,4.76,1051.24,regular,11
5,Veg Maharaja Mac ,306,832,24.17,37.94,16.83,0.28,36.19,93.84,11.52,6.92,1529.22,regular,17
6,Green Chilli Aloo Naan p,132  ,356,7.91,15.08,6.11,0.24,9.45,46.36,4.53,1.15,579.6,regular,1
7,Pizza Puff ,87 ,228,5.45,11.44,5.72,0.09,5.17,24.79,2.73,0.35,390.74,regular,20
8,Mc chicken Burger , 173 ,400,15.66,15.7,5.47,0.16,31.17,47.98,5.53,4.49,766.33,regular,7
9,FILLET-O-FISH Burger , 136 ,348,15.44,14.16,5.79,0.21,32.83,38.85,5.88,3.54,530.54,regular,6
10, Mc Spicy Chicken Burger ,186 ,451,21.46,19.36,7.63,0.18,66.04,46.08,2.52,4.49,928.52,regular,19
11, Spicy Chicken Wrap , 257 ,567,23.74,26.89,12.54,0.27,87.63,57.06,8.92,1.08,1152.38,regular,8
12, Chicken Maharaja Mac , 296  ,689,34,36.69,10.33,0.25,81.49,55.39,7.48,6.14,1854.71,regular,9
13,American Chicken Burger ,165  ,446,20.29,22.94,7.28,0.15,47.63,38.54,5.08,4.76,1132.3,regular,6
14,Chicken Kebab Burger ,138  ,357,8.64,14.02,4.84,0.13,1.51,47.9,3.64,3.49,548.79,regular,9
15,Green Chilli Kebab naan,138,230,5.67,9.32,3.27,0.19,8.74,31.06,4.89,1.15,410.78,regular,6
16,Mc Egg Masala Burger, 126.2 ,290,12.45,12.27,3.64,0.11,213.09,32.89,4.66,3.64,757.91,regular,1
17,Mc Egg Burger for Happy Meal,123,282,12.29,12.21,3.63,0.11,213.09,31.32,3.28,3.64,399.41,regular,5
18,Ghee Rice with Mc Spicy Fried Chicken 1 pc,325,720,26.91,29.2,5.08,0.3,31.32,77.47,0.58,0.35,2399.49,regular,8
19,McSpicy Fried Chicken 1 pc,115,248,17.33,14.29,2.82,0.06,31.11,12.7,0.32,0,873.89,regular,3
20,4 piece Chicken McNuggets ,64,169,10.03,9.54,4.45,0.1,24.66,10.5,0.72,0,313.25,regular,7
21,6 piece Chicken McNuggets ,96,254,15.04,14.3,6.68,0.14,36.99,15.74,0.29,0,469.87,regular,2
22,9 piece Chicken McNuggets ,144 ,381,22.56,21.46,10.02,0.06,55.48,23.62,0.44,0,704.81,regular,7
23,2 piece Chicken Strips ,58 ,164,10.17,12.38,11.41,0.09,30.1,2.68,0.72,0,477.22,regular,1
24,3 piece Chicken Strips , 87 ,246,15.26,18.57,17.12,75.26,45.15,4.02,0.39,0,715.83,regular,19
25,5 piece Chicken Strips , 145 ,411,25.43,28.54,0.15,0.08,6.7,0.73,0.55,0,1193.052,regular,8"""

# Data Loading and Preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv(io.StringIO(CSV_CONTENT), on_bad_lines='skip', delimiter=',')
    
    # Data preprocessing
    numeric_cols = ['calories', 'protien', 'totalfat', 'carbs']
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    
    df['category'] = df['menu'].apply(lambda x: 'Healthy' if x in ['regular', 'breakfast'] else 'Treat')
    
    return df

df = load_data()

# Generate synthetic user preferences for collaborative filtering
@st.cache_data
def generate_user_preferences():
    np.random.seed(42)
    num_users = 100
    user_preferences = pd.DataFrame(index=range(num_users), columns=df.index)
    
    for user in range(num_users):
        # Assign random preferences (ratings) to 20% of meals
        for meal in np.random.choice(df.index, size=int(0.2 * len(df)), replace=False):
            user_preferences.loc[user, meal] = np.random.randint(1, 6)
    
    # Fill NaN with 0 (not rated)
    return user_preferences.fillna(0)

user_preferences = generate_user_preferences()

def calculate_bmi(weight, height):
    if height == 0:
        return 0
    return weight / ((height / 100) ** 2)

# Content-Based Filtering with dynamic recommendation count
def content_based_recommendations(bmi, df):
    df_norm = df.copy()
    # Normalize features
    for feature in ['calories', 'protien', 'totalfat', 'carbs']:
        df_norm[feature] = (df_norm[feature] - df_norm[feature].min()) / \
                          (df_norm[feature].max() - df_norm[feature].min())
    
    # Dynamic weight calculation based on BMI
    calorie_weight = np.clip(1 - (bmi / 40), 0, 1)
    protein_weight = np.clip((bmi / 40), 0, 1)
    fat_weight = np.clip(1 - abs(22 - bmi) / 40, 0, 1)
    carbs_weight = np.clip(1 - (bmi / 40), 0, 1)
    
    # Normalize weights to sum to 1
    weights = np.array([calorie_weight, protein_weight, fat_weight, carbs_weight])
    user_profile = weights / weights.sum()
    
    # Filter based on BMI
    calorie_mean = df['calories'].mean()
    calorie_std = df['calories'].std()
    
    if bmi < 18.5:
        filtered = df[df['calories'] >= calorie_mean + (bmi/18.5)*calorie_std]
        recommendation_type = f"High-Calorie Recommendations (BMI {bmi:.1f})"
        num_rec = 6
    elif 18.5 <= bmi < 25:
        filtered = df[(df['calories'] >= calorie_mean - ((bmi-18.5)/6.5)*calorie_std) &
                     (df['calories'] <= calorie_mean + ((25-bmi)/5)*calorie_std)]
        recommendation_type = f"Balanced Recommendations (BMI {bmi:.1f})"
        num_rec = 4
    else:
        filtered = df[df['calories'] <= calorie_mean - ((bmi-25)/15)*calorie_std]
        recommendation_type = f"Low-Calorie Recommendations (BMI {bmi:.1f})"
        num_rec = 3
    
    # Calculate similarities
    similarities = cosine_similarity([user_profile], df_norm[['calories', 'protien', 'totalfat', 'carbs']].values)
    filtered_indices = filtered.index.tolist()
    filtered_scores = [(i, similarities[0][i]) for i in filtered_indices]
    top_indices = [i[0] for i in sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:num_rec]]
    
    return df.loc[top_indices], recommendation_type, num_rec

# Collaborative Filtering with BMI-based noise
def collaborative_filtering(user_preferences, content_recs, num_recs, bmi):
    cb_indices = content_recs.index.tolist()
    
    # Create synthetic current user profile
    current_user = pd.Series(0, index=user_preferences.columns)
    current_user[cb_indices] = 5  # Boost content-based recommendations
    
    # Add BMI-based noise
    bmi_noise = np.random.normal(loc=bmi/40, scale=0.1, size=len(current_user))
    current_user = current_user + bmi_noise
    
    # Find similar users
    user_similarities = cosine_similarity([current_user.values], user_preferences.values)[0]
    similar_users = user_preferences.iloc[np.argsort(user_similarities)[-10:]]  # Top 10 similar users
    
    # Aggregate recommendations
    collab_recs = pd.Series(0, index=user_preferences.columns)
    for _, user in similar_users.iterrows():
        collab_recs += user
    
    # Remove content-based recs and select top N
    collab_recs[cb_indices] = 0
    top_indices = collab_recs.nlargest(num_recs).index.tolist()
    
    return df.loc[top_indices]

# Visualization functions
def show_radar_chart(selected_meals):
    categories = ['calories', 'protien', 'totalfat', 'carbs']
    fig = go.Figure()
    
    for _, meal in selected_meals.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[meal[c] for c in categories],
            theta=categories,
            fill='toself',
            name=meal['item']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# PDF Generation
def generate_diet_plan_pdf(user_name, bmi, content_recs, collab_recs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"{user_name}'s Diet Plan (BMI: {bmi:.1f})", ln=True, align='C')
    
    # Content-based recommendations
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Primary Recommendations:", ln=True)
    for _, row in content_recs.iterrows():
        pdf.multi_cell(0, 8, f"- {row['item']}: {row['calories']} cals, {row['protien']}g protein")
    
    # Collaborative recommendations
    pdf.ln(10)
    pdf.cell(200, 10, txt="Additional Suggestions:", ln=True)
    for _, row in collab_recs.iterrows():
        pdf.multi_cell(0, 8, f"- {row['item']}: {row['calories']} cals, {row['protien']}g protein")
    
    pdf_file = f"diet_plan_{user_name}.pdf"
    pdf.output(pdf_file)
    return pdf_file

# CSS Styling
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Main App
st.title("🍔 Dynamic Meal Recommender 🥗")

# Navigation Buttons
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
with col_nav1:
    if st.button("← Home", use_container_width=True):
        switch_page("HomePage")
with col_nav3:
    if st.button("History →", use_container_width=True):
        switch_page("Visualizations")

# Personal Details Section
with st.container():
    st.header("Personal Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        weight = st.number_input('Weight (kg):⚖️', min_value=30.0, value=70.0)
    with col2:
        height = st.number_input('Height (cm):📏', min_value=100.0, value=170.0)
    with col3:
        age = st.number_input('Age:🎂', min_value=1, value=25)
    with col4:
        gender = st.selectbox("Gender:🚻", ["Male", "Female", "Other"])
    
    bmi = calculate_bmi(weight, height)
    st.markdown(f"""
    <div class="bmi-result">
        Your BMI: {bmi:.1f}<br>
        {["Underweight 🤢", "Normal Weight 😊", "Overweight ⚠️"][
            (bmi >= 18.5) + (bmi >= 25)
        ]}
    </div>
    """, unsafe_allow_html=True)

# Recommendation Generation
if st.button("Generate Meal Recommendations 🍽️"):
    content_recs, rec_type, num_recs = content_based_recommendations(bmi, df)
    collab_recs = collaborative_filtering(user_preferences, content_recs, num_recs, bmi)
    
    # Store in session state
    st.session_state.update({
        'content_recs': content_recs,
        'collab_recs': collab_recs,
        'bmi': bmi,
        'rec_history': st.session_state.get('rec_history', []) + [{
            'bmi': bmi,
            'recommendations': content_recs['item'].tolist()
        }]
    })

# Display Recommendations
if 'content_recs' in st.session_state:
    st.header(st.session_state.rec_type)
    
    # Content-based Recommendations in Cards
    cols = st.columns(3)
    for idx, (_, row) in enumerate(st.session_state.content_recs.iterrows()):
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
    
    # Collaborative Recommendations
    st.header("You Might Also Like")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(st.session_state.collab_recs.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="meal-card collab">
                <h4>{row['item']}</h4>
                <p>Calories: {row['calories']:.0f}</p>
                <p>Protein: {row['protien']}g</p>
                <p>Carbs: {row['carbs']}g</p>
                <p>Fat: {row['totalfat']}g</p>
                <p>Rating: {row['Ratings']}/20</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations
    st.header("Nutritional Profile Comparison")
    show_radar_chart(st.session_state.content_recs)
    
    # PDF Generation
    st.header("Export Options")
    user_name = st.text_input("Enter your name for PDF export:")
    if st.button("Generate PDF Report"):
        if user_name:
            pdf_file = generate_diet_plan_pdf(
                user_name,
                st.session_state.bmi,
                st.session_state.content_recs,
                st.session_state.collab_recs
            )
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "Download Diet Plan",
                    data=f,
                    file_name=pdf_file,
                    mime="application/octet-stream"
                )
        else:
            st.warning("Please enter your name to generate PDF")

# Recommendation History Sidebar
st.sidebar.header("Recommendation History")
if 'rec_history' in st.session_state:
    for rec in reversed(st.session_state.rec_history[-5:]):
        with st.sidebar.expander(f"BMI {rec['bmi']:.1f}"):
            for meal in rec['recommendations']:
                st.write(f"- {meal}")
else:
    st.sidebar.write("No history yet")
