import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import random
from streamlit_extras.switch_page_button import switch_page

# Set page config
st.set_page_config(page_title="Meal Recommender", layout="wide")

# Load CSS
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Dark Mode Toggle (Optional if you implemented it earlier)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
dark_mode_toggle = st.sidebar.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = dark_mode_toggle

# Dataset
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

@st.cache_data
def load_data():
    df = pd.read_csv(io.StringIO(CSV_CONTENT), on_bad_lines='skip', delimiter=',')
    df['calories'] = df['calories'].astype(float)
    df['protien'] = df['protien'].astype(float)
    df['totalfat'] = df['totalfat'].astype(float)
    df['carbs'] = df['carbs'].astype(float)
    df['category'] = df['menu'].apply(lambda x: 'Healthy' if x in ['regular', 'breakfast'] else 'Treat')

    np.random.seed(42)
    num_users = 100
    user_preferences = pd.DataFrame(index=range(num_users), columns=df.index)

    for user in range(num_users):
        for meal in np.random.choice(df.index, size=int(0.2 * len(df)), replace=False):
            user_preferences.loc[user, meal] = np.random.randint(1, 6)
    user_preferences = user_preferences.fillna(0)

    return df, user_preferences

df, user_preferences = load_data()

def calculate_bmi(weight, height):
    if height == 0:
        return 0
    return weight / ((height / 100) ** 2)

def content_based_recommendations(bmi, df):
    df_norm = df.copy()
    for feature in ['calories', 'protien', 'totalfat', 'carbs']:
        df_norm[feature] = (df_norm[feature] - df_norm[feature].min()) / (df_norm[feature].max() - df_norm[feature].min())

    features = df_norm[['calories', 'protien', 'totalfat', 'carbs']].values

    if bmi < 18.5:
        user_profile = [0.1, 0.5, 0.2, 0.2]
        filtered = df[df['calories'] >= df['calories'].mean()]
        recommendation_type = "High-Calorie Recommendations for Weight Gain"
        num_recommendations = 5
    elif 18.5 <= bmi < 25:
        user_profile = [0.25, 0.25, 0.25, 0.25]
        filtered = df
        recommendation_type = "Balanced Meal Recommendations"
        num_recommendations = 4
    else:
        user_profile = [0.5, 0.3, 0.1, 0.1]
        filtered = df[df['calories'] <= df['calories'].mean()]
        recommendation_type = "Low-Calorie Recommendations for Weight Management"
        num_recommendations = 3

    similarities = cosine_similarity([user_profile], features)
    indices = filtered.index.tolist()
    similarity_scores = [(i, similarities[0][i]) for i in indices]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in similarity_scores[:num_recommendations]]

    return filtered.loc[top_indices], recommendation_type, num_recommendations

def collaborative_filtering(user_preferences, content_based_recs, num_collab_recs):
    cb_indices = content_based_recs.index.tolist()
    current_user = pd.Series(0, index=user_preferences.columns)
    current_user[cb_indices] = 5

    user_similarities = cosine_similarity([current_user.values], user_preferences.values)[0]
    similar_users = user_preferences.iloc[np.argsort(user_similarities)[-10:]]
    collaborative_recs = pd.Series(0, index=user_preferences.columns)

    for _, user in similar_users.iterrows():
        collaborative_recs += user

    collaborative_recs[cb_indices] = 0
    top_collab_indices = collaborative_recs.nlargest(num_collab_recs).index.tolist()

    return df.loc[top_collab_indices]

def calculate_bmi(weight, height):
    if height == 0:
        return 0  # or you might want to display an error message or prompt the user to enter a valid height
    return weight / ((height/100) ** 2)
# User Input
st.title("Personalized Meal Recommendations")

st.header("Enter Your Personal Details")
col1, col2 = st.columns(2)
with col1:
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=st.session_state.get('weight', 00.0))
    height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=st.session_state.get('height', 00.0))
with col2:
    gender = st.selectbox("Gender", options=["Male", "Female", "Prefer not to say"], index=st.session_state.get('gender_index', 0))
    age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.get('age', 25))

# Save inputs to session_state
st.session_state.weight = weight
st.session_state.height = height
st.session_state.gender = gender
st.session_state.age = age
st.session_state.gender_index = ["Male", "Female", "Prefer not to say"].index(gender)

bmi = calculate_bmi(weight, height)
st.session_state.bmi = bmi
st.subheader(f"Your BMI: {bmi:.1f}")

if bmi < 18.5:
    st.warning("Underweight - Recommending high-calorie meals")
elif 18.5 <= bmi < 25:
    st.success("Normal Weight - Recommending balanced meals")
else:
    st.error("Overweight/Obesity - Recommending low-calorie meals")

if st.button("Generate Recommendations", key="gen_rec"):
    content_recs, recommendation_type, num_recs = content_based_recommendations(bmi, df)
    collaborative_recs = collaborative_filtering(user_preferences, content_recs, num_recs)

    st.session_state.content_recs = content_recs
    st.session_state.collaborative_recs = collaborative_recs
    st.session_state.recommendation_type = recommendation_type
    st.session_state.num_recs = num_recs

    st.header(recommendation_type)
    st.markdown("**Recommended meals based on your nutritional needs:**")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(content_recs.iterrows()):
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

    st.header("You Might Also Like")
    st.markdown("**Recommended based on similar users' preferences:**")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(collaborative_recs.iterrows()):
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

      # Download buttons for exporting recommendations
    st.download_button(
        label="Download Content-Based Recommendations as CSV",
        data=content_recs.to_csv(index=False),
        file_name="content_based_recommendations.csv",
        mime='text/csv'
    )

    st.download_button(
        label="Download Collaborative Recommendations as CSV",
        data=collaborative_recs.to_csv(index=False),
        file_name="collaborative_recommendations.csv",
        mime='text/csv'
    )

    if st.button("View Detailed Visualizations", key="view_viz"):
        switch_page("Visualizations")
