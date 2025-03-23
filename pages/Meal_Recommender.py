import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import io
from streamlit_extras.switch_page_button import switch_page
from fpdf import FPDF

st.set_page_config(page_title="Dynamic Meal Recommender", layout="wide")

# Initialize session state keys
def init_session_state():
    defaults = {
        'rec_type': None,
        'content_recs': None,
        'collaborative_recs': None,
        'bmi': None,
        'rec_history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Load dataset from provided CSV content
CSV_CONTENT = """<-- your CSV content here -->"""

@st.cache_data
def load_data():
    df = pd.read_csv(io.StringIO(CSV_CONTENT), on_bad_lines='skip', delimiter=',')
    numeric_cols = ['calories', 'protien', 'totalfat', 'carbs']
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    df['category'] = df['menu'].apply(lambda x: 'Healthy' if x in ['regular', 'breakfast'] else 'Treat')
    return df

df = load_data()

@st.cache_data
def generate_user_preferences():
    np.random.seed(42)
    user_preferences = pd.DataFrame(index=range(100), columns=df.index)
    for user in range(100):
        for meal in np.random.choice(df.index, size=int(0.2 * len(df)), replace=False):
            user_preferences.loc[user, meal] = np.random.randint(1, 6)
    return user_preferences.fillna(0)

user_preferences = generate_user_preferences()

def calculate_bmi(weight, height):
    return 0 if height == 0 else weight / ((height / 100) ** 2)

def content_based_recommendations(bmi, df):
    df_norm = df.copy()
    for feature in ['calories', 'protien', 'totalfat', 'carbs']:
        df_norm[feature] = (df_norm[feature] - df_norm[feature].min()) / (df_norm[feature].max() - df_norm[feature].min())
    weights = np.array([
        np.clip(1 - (bmi / 40), 0, 1),
        np.clip((bmi / 40), 0, 1),
        np.clip(1 - abs(22 - bmi) / 40, 0, 1),
        np.clip(1 - (bmi / 40), 0, 1)
    ])
    user_profile = weights / weights.sum()
    calorie_mean = df['calories'].mean()
    calorie_std = df['calories'].std()
    if bmi < 18.5:
        filtered = df[df['calories'] >= calorie_mean + (bmi/18.5)*calorie_std]
        rec_type = f"High-Calorie Meals (BMI {bmi:.1f})"
        num_recs = 6
    elif bmi < 25:
        filtered = df[(df['calories'] >= calorie_mean - ((bmi-18.5)/6.5)*calorie_std) & (df['calories'] <= calorie_mean + ((25-bmi)/5)*calorie_std)]
        rec_type = f"Balanced Meals (BMI {bmi:.1f})"
        num_recs = 4
    else:
        filtered = df[df['calories'] <= calorie_mean - ((bmi-25)/15)*calorie_std]
        rec_type = f"Low-Calorie Meals (BMI {bmi:.1f})"
        num_recs = 3
    sims = cosine_similarity([user_profile], df_norm[['calories', 'protien', 'totalfat', 'carbs']].values)
    scores = [(i, sims[0][i]) for i in filtered.index.tolist()]
    top_indices = [i[0] for i in sorted(scores, key=lambda x: x[1], reverse=True)[:num_recs]]
    return df.loc[top_indices], rec_type, num_recs

def collaborative_filtering(user_preferences, content_recs, num_recs, bmi):
    cb_indices = content_recs.index.tolist()
    current_user = pd.Series(0, index=user_preferences.columns)
    current_user[cb_indices] = 5
    current_user += np.random.normal(loc=bmi/40, scale=0.1, size=len(current_user))
    sims = cosine_similarity([current_user.values], user_preferences.values)[0]
    similar_users = user_preferences.iloc[np.argsort(sims)[-10:]]
    collab_recs = pd.Series(0, index=user_preferences.columns)
    for _, user in similar_users.iterrows():
        collab_recs += user
    collab_recs[cb_indices] = 0
    top_indices = collab_recs.nlargest(num_recs).index.tolist()
    return df.loc[top_indices]

def show_radar_chart(selected_meals):
    categories = ['calories', 'protien', 'totalfat', 'carbs']
    fig = go.Figure()
    for _, meal in selected_meals.iterrows():
        fig.add_trace(go.Scatterpolar(r=[meal[c] for c in categories], theta=categories, fill='toself', name=meal['item']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=500)
    st.plotly_chart(fig, use_container_width=True)

def generate_diet_plan_pdf(user_name, bmi, content_recs, collab_recs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=f"{user_name}'s Diet Plan (BMI: {bmi:.1f})", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Primary Recommendations:", ln=True)
    for _, row in content_recs.iterrows():
        pdf.multi_cell(0, 8, f"- {row['item']}: {row['calories']} cals, {row['protien']}g protein")
    pdf.ln(10)
    pdf.cell(200, 10, txt="Additional Suggestions:", ln=True)
    for _, row in collab_recs.iterrows():
        pdf.multi_cell(0, 8, f"- {row['item']}: {row['calories']} cals, {row['protien']}g protein")
    pdf_file = f"diet_plan_{user_name}.pdf"
    pdf.output(pdf_file)
    return pdf_file

with open('style1.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🍔 Dynamic Meal Recommender 🥗")

col_nav1, col_nav3 = st.columns([1, 1])
with col_nav1:
    if st.button("← Home", use_container_width=True):
        switch_page("HomePage")
with col_nav3:
    if st.button("Visualizations →", use_container_width=True):
        switch_page("Visualizations")

with st.container():
    st.header("Personal Information")
    col1, col2, col3, col4 = st.columns(4)
    weight = col1.number_input('Weight (kg):⚖️', min_value=30.0, value=70.0)
    height = col2.number_input('Height (cm):📏', min_value=100.0, value=170.0)
    age = col3.number_input('Age:🎂', min_value=1, value=25)
    gender = col4.selectbox("Gender:🚻", ["Male", "Female", "Other"])
    bmi = calculate_bmi(weight, height)
    st.markdown(f"""
    <div class="bmi-result">
        Your BMI: {bmi:.1f}<br>
        {['Underweight 🤢', 'Normal Weight 😊', 'Overweight ⚠️'][ (bmi >= 18.5) + (bmi >= 25) ]}
    </div>
    """, unsafe_allow_html=True)

if st.button("Generate Recommendations"):
    content_recs, rec_type, num_recs = content_based_recommendations(bmi, df)
    collaborative_recs = collaborative_filtering(user_preferences, content_recs, num_recs, bmi)
    st.session_state.update({
        'content_recs': content_recs,
        'collaborative_recs': collaborative_recs,
        'rec_type': rec_type,
        'bmi': bmi,
        'rec_history': st.session_state.rec_history + [{
            'bmi': bmi,
            'recommendations': content_recs['item'].tolist()
        }]
    })
    st.success("✅ Recommendations generated! Check below.")

if st.session_state.content_recs is not None:
    st.header(st.session_state.rec_type)
    st.subheader("🍱 Content-Based Recommendations")
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

if st.session_state.collaborative_recs is not None:
    st.header("You Might Also Like")
    cols = st.columns(3)
    for idx, (_, row) in enumerate(st.session_state.collaborative_recs.iterrows()):
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
    st.header("Nutritional Profile Comparison")
    show_radar_chart(st.session_state.content_recs)

    st.header("Export Options")
    user_name = st.text_input("Enter your name for PDF export:")
    if st.button("Generate PDF Report") and user_name:
        pdf_file = generate_diet_plan_pdf(user_name, bmi, st.session_state.content_recs, st.session_state.collaborative_recs)
        with open(pdf_file, "rb") as f:
            st.download_button("Download Diet Plan", data=f, file_name=pdf_file, mime="application/pdf")

st.sidebar.header("Recommendation History")
if st.session_state.rec_history:
    for rec in reversed(st.session_state.rec_history[-5:]):
        with st.sidebar.expander(f"BMI {rec['bmi']:.1f}"):
            for meal in rec['recommendations']:
                st.write(f"- {meal}")
else:
    st.sidebar.write("No history yet.")
