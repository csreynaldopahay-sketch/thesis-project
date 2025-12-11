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

# Define feature columns (must match the order used during training)
FEATURE_COLS = [
    "ampicillin_encoded", "amoxicillin/clavulanic_acid_encoded", "ceftaroline_encoded",
    "cefalexin_encoded", "cefalotin_encoded", "cefpodoxime_encoded", "cefotaxime_encoded",
    "cefovecin_encoded", "ceftiofur_encoded", "ceftazidime/avibactam_encoded",
    "imepenem_encoded", "amikacin_encoded", "gentamicin_encoded", "neomycin_encoded",
    "nalidixic_acid_encoded", "enrofloxacin_encoded", "marbofloxacin_encoded",
    "pradofloxacin_encoded", "doxycycline_encoded", "tetracycline_encoded",
    "nitrofurantoin_encoded", "chloramphenicol_encoded", "trimethoprim/sulfamethazole_encoded"
]

# Define antibiotics for display (matching the feature columns order)
ANTIBIOTICS = [
    "ampicillin", "amoxicillin/clavulanic_acid", "ceftaroline", "cefalexin",
    "cefalotin", "cefpodoxime", "cefotaxime", "cefovecin", "ceftiofur",
    "ceftazidime/avibactam", "imepenem", "amikacin", "gentamicin", "neomycin",
    "nalidixic_acid", "enrofloxacin", "marbofloxacin", "pradofloxacin",
    "doxycycline", "tetracycline", "nitrofurantoin", "chloramphenicol",
    "trimethoprim/sulfamethazole"
]

# Encoding map for resistance values
ENCODING_MAP = {'s': 0, 'i': 1, 'r': 2, 'n': 0}  # 'n' (not tested) defaults to susceptible

# Model name mapping
MODEL_NAME_MAP = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes"
}


@st.cache_resource
def load_models(model_type):
    """Load trained models from disk."""
    models_dir = Path("models")
    mar_model = None
    species_model = None
    
    mar_path = models_dir / f"mar_{model_type}.pkl"
    species_path = models_dir / f"species_{model_type}.pkl"
    
    if mar_path.exists():
        mar_model = joblib.load(mar_path)
    
    if species_path.exists():
        species_model = joblib.load(species_path)
    
    return mar_model, species_model


def preprocess_input(resistance_profile):
    """Convert resistance profile to feature array for model prediction."""
    features = []
    for antibiotic in ANTIBIOTICS:
        value = resistance_profile.get(antibiotic, 's').lower()
        features.append(ENCODING_MAP.get(value, 0))
    return np.array(features).reshape(1, -1)


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

# Load models based on selection
model_type = MODEL_NAME_MAP[model_option]
mar_model, species_model = load_models(model_type)

# Show model status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status:**")
st.sidebar.write(f"MDR Model: {'✅ Loaded' if mar_model else '❌ Not found'}")
st.sidebar.write(f"Species Model: {'✅ Loaded' if species_model else '❌ Not found'}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📂 File Upload", "📊 Results Visualization"])

with tab1:
    st.header("Enter Resistance Profile")
    
    col1, col2, col3 = st.columns(3)
    
    resistance_profile = {}
    
    for i, antibiotic in enumerate(ANTIBIOTICS):
        with [col1, col2, col3][i % 3]:
            value = st.selectbox(
                antibiotic.replace("_", " ").replace("/", " / ").title(),
                options=["S (Susceptible)", "I (Intermediate)", "R (Resistant)", "N (Not Tested)"],
                key=f"ab_{antibiotic}"
            )
            resistance_profile[antibiotic] = value[0].lower()
    
    if st.button("🔮 Predict", type="primary"):
        st.subheader("Prediction Results")
        
        # Preprocess input
        X = preprocess_input(resistance_profile)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if mar_model is not None:
                mar_pred = mar_model.predict(X)[0]
                mar_label = "High MAR (MDR)" if mar_pred == 1 else "Low MAR"
                
                if hasattr(mar_model, 'predict_proba'):
                    mar_proba = mar_model.predict_proba(X)[0]
                    mar_confidence = max(mar_proba) * 100
                    st.metric(
                        label="MDR Prediction",
                        value=mar_label,
                        delta=f"{mar_confidence:.1f}% confidence"
                    )
                else:
                    st.metric(
                        label="MDR Prediction",
                        value=mar_label,
                        delta="Based on resistance profile"
                    )
            else:
                st.warning("MDR model not loaded. Please check if model files exist in 'models/' directory.")
        
        with col2:
            if species_model is not None:
                species_pred = species_model.predict(X)[0]
                
                if hasattr(species_model, 'predict_proba'):
                    species_proba = species_model.predict_proba(X)[0]
                    species_confidence = max(species_proba) * 100
                    st.metric(
                        label="Predicted Species",
                        value=str(species_pred).replace("_", " ").title(),
                        delta=f"{species_confidence:.1f}% confidence"
                    )
                    
                    # Show top 3 predictions
                    st.markdown("**Top 3 Predictions:**")
                    classes = species_model.classes_
                    top_indices = np.argsort(species_proba)[-3:][::-1]
                    for idx in top_indices:
                        species_name = str(classes[idx]).replace("_", " ").title()
                        prob = species_proba[idx] * 100
                        st.write(f"- {species_name}: {prob:.1f}%")
                else:
                    st.metric(
                        label="Predicted Species",
                        value=str(species_pred).replace("_", " ").title(),
                        delta="Based on resistance profile"
                    )
            else:
                st.warning("Species model not loaded. Please check if model files exist in 'models/' directory.")

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
            if mar_model is None and species_model is None:
                st.error("No models loaded. Please check if model files exist.")
            else:
                # Try to extract features from uploaded data
                results = []
                for idx, row in df.iterrows():
                    profile = {}
                    for antibiotic in ANTIBIOTICS:
                        # Try different column name formats
                        col_name = f"{antibiotic}_int"
                        if col_name in row:
                            profile[antibiotic] = str(row[col_name]).lower()
                        elif antibiotic in row:
                            profile[antibiotic] = str(row[antibiotic]).lower()
                        else:
                            profile[antibiotic] = 's'  # Default to susceptible
                    
                    X = preprocess_input(profile)
                    result = {'index': idx}
                    
                    if mar_model is not None:
                        mar_pred = mar_model.predict(X)[0]
                        result['mdr_prediction'] = "High MAR" if mar_pred == 1 else "Low MAR"
                    
                    if species_model is not None:
                        species_pred = species_model.predict(X)[0]
                        result['species_prediction'] = str(species_pred)
                    
                    results.append(result)
                
                results_df = pd.DataFrame(results)
                st.success(f"Predictions completed for {len(results)} samples!")
                st.dataframe(results_df)

with tab3:
    st.header("Results Visualization")
    
    # Sample visualization
    st.subheader("Resistance Pattern Distribution")
    
    # Create sample data for visualization
    sample_data = pd.DataFrame({
        'Antibiotic': [a.replace("_", " ").replace("/", "/\\n").title() for a in ANTIBIOTICS[:10]],
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
