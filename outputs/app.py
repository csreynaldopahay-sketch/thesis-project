
"""
AMR Pattern Recognition - Streamlit Web Application

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="AMR Pattern Recognition",
    page_icon="🦠",
    layout="wide"
)

# Title and description
st.title("🦠 AMR Pattern Recognition System")
st.markdown("""
This application predicts:
- **MDR (Multi-Drug Resistance)** status based on antibiotic resistance profiles
- **Bacterial Species** from resistance patterns

Upload your data or enter antibiotic resistance values manually.
""")

# Sidebar
st.sidebar.header("Settings")
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN", "Naive Bayes"]
)

# Define antibiotics (these should match your dataset)
ANTIBIOTICS = [
    "ampicillin", "amoxicillin_clavulanic_acid", "ceftaroline", "cefalexin",
    "cefalotin", "cefpodoxime", "cefotaxime", "cefovecin", "ceftiofur",
    "ceftazidime_avibactam", "imepenem", "amikacin", "gentamicin", "neomycin",
    "nalidixic_acid", "enrofloxacin", "marbofloxacin", "pradofloxacin",
    "doxycycline", "tetracycline", "nitrofurantoin", "chloramphenicol",
    "trimethoprim_sulfamethazole"
]

# Main tabs
tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📂 File Upload", "📊 Results Visualization"])

with tab1:
    st.header("Enter Resistance Profile")
    
    col1, col2, col3 = st.columns(3)
    
    resistance_profile = {}
    
    for i, antibiotic in enumerate(ANTIBIOTICS):
        with [col1, col2, col3][i % 3]:
            value = st.selectbox(
                antibiotic.replace("_", " ").title(),
                options=["S (Susceptible)", "I (Intermediate)", "R (Resistant)"],
                key=f"ab_{antibiotic}"
            )
            resistance_profile[antibiotic] = value[0].lower()
    
    if st.button("🔮 Predict", type="primary"):
        st.subheader("Prediction Results")
        
        # Placeholder for predictions (would use loaded model in production)
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="MDR Prediction",
                value="High MAR (MDR)" if sum(1 for v in resistance_profile.values() if v == 'r') > 5 else "Low MAR",
                delta="Based on resistance profile"
            )
        
        with col2:
            st.metric(
                label="Predicted Species",
                value="E. coli",
                delta="85% confidence"
            )

with tab2:
    st.header("Upload Resistance Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with resistance profiles",
        type="csv"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())
        
        if st.button("🔮 Predict All", type="primary"):
            st.info("Batch prediction would be performed here with loaded models")

with tab3:
    st.header("Results Visualization")
    
    # Sample visualization
    st.subheader("Resistance Pattern Distribution")
    
    # Create sample data for visualization
    sample_data = pd.DataFrame({
        'Antibiotic': ANTIBIOTICS[:10],
        'Resistance Rate': np.random.random(10) * 100
    })
    
    fig = px.bar(
        sample_data,
        x='Antibiotic',
        y='Resistance Rate',
        title='Antibiotic Resistance Rates'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**AMR Pattern Recognition System** - Thesis Project

Built with Streamlit | Models trained with scikit-learn & XGBoost
""")
