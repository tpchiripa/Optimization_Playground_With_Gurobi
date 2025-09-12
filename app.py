import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("random_forest_player_model.pkl")

# Features used during training
trained_features = ['age', 'height', 'weight', 'potential', 'short_passing', 
                    'long_passing', 'ball_control', 'dribbling', 'stamina', 
                    'strength', 'acceleration', 'positioning', 'marking', 
                    'standing_tackle', 'sliding_tackle', 'crossing', 'finishing',
                    'heading_accuracy', 'interceptions', 'volleys', 'curve', 'free_kick_accuracy']

st.set_page_config(page_title="⚽ Player Market Value Predictor", layout="wide")

st.title("⚽ Player Market Value Predictor")
st.markdown("Predict a player's overall rating (proxy for market value) with interactive sliders and uploaded data.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Interactive Prediction", "Live Player Comparison", "Feature Importance"])

# --- Tab 1: Interactive Sliders ---
with tab1:
    st.subheader("Interactive Player Prediction")

    # Create sliders for numeric features
    input_data = {}
    for feat in ['age', 'height', 'weight', 'potential', 'short_passing', 'long_passing',
                 'ball_control', 'dribbling', 'stamina', 'strength', 'acceleration',
                 'positioning', 'marking', 'standing_tackle', 'sliding_tackle',
                 'crossing', 'finishing', 'heading_accuracy', 'interceptions', 
                 'volleys', 'curve', 'free_kick_accuracy']:
        min_val = 0
        max_val = 100 if feat not in ['age', 'height', 'weight'] else 40 if feat=='age' else 250 if feat=='weight' else 220
        default_val = 50
        if feat=='age':
            default_val=25
        elif feat=='height':
            default_val=175
        elif feat=='weight':
            default_val=70
        input_data[feat] = st.slider(feat.replace('_',' ').title(), min_value=min_val, max_value=max_val, value=default_val)

    # Predict button
    if st.button("Predict Rating"):
        player_df = pd.DataFrame([input_data])
        predicted_rating = model.predict(player_df[trained_features])[0]
        st.success(f"Predicted Overall Rating: {predicted_rating:.2f}")

# --- Tab 2: Live Player Comparison ---
with tab2:
    st.subheader("Live Player Comparison")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file with player attributes", type=['csv', 'xlsx'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            new_players = pd.read_csv(uploaded_file)
        else:
            new_players = pd.read_excel(uploaded_file)
        
        # Predict ratings
        new_players['Predicted Rating'] = model.predict(new_players[trained_features])
        
        # Sort by rating
        new_players.sort_values(by='Predicted Rating', ascending=False, inplace=True)
        
        # Display styled dataframe
        st.dataframe(
            new_players.style.background_gradient(cmap='PuBu', axis=0),
            use_container_width=True
        )
    else:
        st.info("Upload a CSV or Excel file to see predictions and compare players live.")

# --- Tab 3: Feature Importance ---
with tab3:
    st.subheader("Top 10 Features in Model")
    
    feature_importances = pd.Series(model.feature_importances_, index=trained_features)
    top10_features = feature_importances.sort_values(ascending=False).head(10)
    
    fig = px.bar(
        x=top10_features.values,
        y=top10_features.index,
        orientation='h',
        labels={'x':'Importance', 'y':'Feature'},
        title='Top 10 Features Driving Overall Rating'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("App created by **Tichaona Chiripa**")
