import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.switch_page_button import switch_page

# Load data from Meal_Recommender.py
if 'content_recs' not in st.session_state:
    st.session_state.content_recs = None
if 'collaborative_recs' not in st.session_state:
    st.session_state.collaborative_recs = None
if 'recommendation_type' not in st.session_state:
    st.session_state.recommendation_type = None
if 'bmi' not in st.session_state:
    st.session_state.bmi = None

# Set up the page
st.set_page_config(page_title="Meal Visualizations", layout="wide")

# Load CSS
with open('style1.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.title("Nutrition Insights & Visualizations")

# Navigation Buttons
col_nav1, col_nav2 = st.columns([1, 1])
with col_nav1:
    if st.button("← Home", use_container_width=True):
        switch_page("HomePage")
with col_nav2:
    if st.button("← Recommendations", use_container_width=True):
        switch_page("Meal_Recommender")

# Check if recommendations exist
if st.session_state.content_recs is None:
    st.warning("No recommendations generated yet. Please go to the Meal Recommender page first.")
    if st.button("Go to Meal Recommender", use_container_width=True):
        switch_page("Meal_Recommender")
else:
    # Get recommendations
    content_recs = st.session_state.content_recs
    collaborative_recs = st.session_state.collaborative_recs
    recommendation_type = st.session_state.recommendation_type
    bmi = st.session_state.bmi
    
    # Visualization Description
    st.markdown("""
    <div class="viz-description">
        <h3>Understanding Your Visualizations</h3>
        <p>These visualizations provide detailed insights into your meal recommendations based on your BMI of {:.1f}. The charts help you understand the nutritional composition of recommended meals and how they compare to your needs.</p>
    </div>
    """.format(bmi), unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Calorie Distribution", "Nutrition Breakdown", "Comparative Analysis"])
    
    with tab1:
        st.header("Calorie Distribution")
        
        # Create bar chart of calories for all recommendations
        all_recs = pd.concat([content_recs, collaborative_recs])
        
        fig = px.bar(
            all_recs,
            x='item',
            y='calories',
            color='category',
            title='Calorie Content of Recommended Meals',
            labels={'item': 'Meal', 'calories': 'Calories', 'category': 'Meal Type'},
            color_discrete_map={'Healthy': '#32CD32', 'Treat': '#FF7F50'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Meal",
            yaxis_title="Calories",
            legend_title="Meal Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Description
        st.markdown("""
        <div class="viz-info">
            <h4>Calorie Distribution Insights:</h4>
            <p>This bar chart shows the calorie content of all recommended meals. The colors indicate whether the meal is categorized as 'Healthy' or 'Treat'. This visualization helps you identify which meals are higher in calories and may be more appropriate for weight gain or which are lower in calories for weight management.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Nutrition Breakdown")
        
        # Create stacked bar chart of macronutrients
        macro_df = content_recs.copy()
        macro_df['meal_index'] = range(len(macro_df))
        
        # Melt the dataframe for stacked bar chart
        melted = pd.melt(
            macro_df,
            id_vars=['meal_index', 'item'],
            value_vars=['protien', 'carbs', 'totalfat'],
            var_name='nutrient',
            value_name='grams'
        )
        
        # Map nutrient names to better labels
        melted['nutrient'] = melted['nutrient'].map({
            'protien': 'Protein',
            'carbs': 'Carbohydrates',
            'totalfat': 'Fat'
        })
        
        # Create stacked bar chart
        fig = px.bar(
            melted,
            x='item',
            y='grams',
            color='nutrient',
            title='Macronutrient Composition of Recommended Meals',
            labels={'item': 'Meal', 'grams': 'Grams', 'nutrient': 'Nutrient'},
            color_discrete_map={'Protein': '#4CAF50', 'Carbohydrates': '#FFC107', 'Fat': '#FF5722'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="Meal",
            yaxis_title="Grams",
            legend_title="Nutrient"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart for average macronutrient distribution
        avg_macros = {
            'Protein': content_recs['protien'].mean(),
            'Carbohydrates': content_recs['carbs'].mean(),
            'Fat': content_recs['totalfat'].mean()
        }
        
        fig = px.pie(
            values=list(avg_macros.values()),
            names=list(avg_macros.keys()),
            title='Average Macronutrient Distribution',
            color_discrete_map={'Protein': '#4CAF50', 'Carbohydrates': '#FFC107', 'Fat': '#FF5722'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Description
        st.markdown("""
        <div class="viz-info">
            <h4>Nutrition Breakdown Insights:</h4>
            <p>The stacked bar chart shows the macronutrient composition of each recommended meal, breaking down the amount of protein, carbohydrates, and fat in grams. This helps you understand the nutritional balance of each meal.</p>
            <p>The pie chart displays the average macronutrient distribution across all recommended meals, giving you a quick overview of the overall nutritional balance in your meal plan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.header("Comparative Analysis")
        
        # Create line chart comparing calories and protein
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=content_recs['item'],
            y=content_recs['calories'],
            mode='lines+markers',
            name='Calories',
            line=dict(color='#FF4B4B', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=content_recs['item'],
            y=content_recs['protien'] * 10,  # Scale protein to be comparable to calories
            mode='lines+markers',
            name='Protein (x10)',
            line=dict(color='#4CAF50', width=2)
        ))
        
        fig.update_layout(
            title='Calories vs. Protein Content',
            xaxis_title='Meal',
            yaxis_title='Value',
            xaxis_tickangle=-45,
            legend_title='Nutrient'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a histogram of protein content
        fig = px.histogram(
            content_recs,
            x='protien',
            nbins=10,
            title='Distribution of Protein Content',
            labels={'protien': 'Protein (g)'},
            color_discrete_sequence=['#4CAF50']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Description
        st.markdown("""
        <div class="viz-info">
            <h4>Comparative Analysis Insights:</h4>
            <p>The line chart compares the calorie and protein content across all recommended meals. Protein values are scaled (multiplied by 10) to allow for easier comparison with calories. This visualization helps you identify meals that offer a good balance of calories and protein based on your needs.</p>
            <p>The histogram shows the distribution of protein content across all recommended meals, helping you understand the range and frequency of protein levels in your meal plan.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary section
    st.header("Summary of Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Nutrition Metrics")
        metrics = {
            "Average Calories": f"{content_recs['calories'].mean():.1f} kcal",
            "Average Protein": f"{content_recs['protien'].mean():.1f} g",
            "Average Carbs": f"{content_recs['carbs'].mean():.1f} g",
            "Average Fat": f"{content_recs['totalfat'].mean():.1f} g",
            "Highest Rated Meal": content_recs.loc[content_recs['Ratings'].idxmax()]['item'],
            "Lowest Calorie Meal": content_recs.loc[content_recs['calories'].idxmin()]['item'],
            "Highest Protein Meal": content_recs.loc[content_recs['protien'].idxmax()]['item']
        }
        
        for metric, value in metrics.items():
            st.markdown(f"**{metric}:** {value}")
    
    with col2:
        st.subheader("Recommendation Insights")
        
        # Recommendation insights based on BMI
        if bmi < 18.5:
            st.markdown("""
            <div class="insight-box underweight">
                <h4>Underweight Recommendations</h4>
                <p>Your meal plan focuses on higher calorie options with increased protein to support healthy weight gain. The visualizations show that recommended meals have higher calorie content while maintaining nutritional balance.</p>
            </div>
            """, unsafe_allow_html=True)
        elif 18.5 <= bmi < 25:
            st.markdown("""
            <div class="insight-box normal">
                <h4>Normal Weight Recommendations</h4>
                <p>Your meal plan offers a balance of nutrients with moderate calorie content to maintain your healthy weight. The visualizations show a good distribution of all macronutrients across recommended meals.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box overweight">
                <h4>Weight Management Recommendations</h4>
                <p>Your meal plan focuses on lower calorie options with adequate protein to support weight management. The visualizations show that recommended meals have controlled calorie content while ensuring nutritional adequacy.</p>
            </div>
            """, unsafe_allow_html=True)

