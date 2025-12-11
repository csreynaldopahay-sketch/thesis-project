"""
PHASE 4: DEPLOYMENT

This module provides:
- Streamlit web application for AMR prediction
- FastAPI REST API endpoints
- Model loading and prediction pipeline
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


class AMRPredictionPipeline:
    """
    Prediction pipeline for AMR analysis.
    
    Handles:
    - Model loading
    - Input preprocessing
    - MDR/MAR prediction (MAR > 0.17 threshold)
    - Species prediction
    - Feature extraction
    
    MDR Definition:
    - A bacterium is considered multi-drug resistant (MDR) if its MAR index > 0.17
    - This aligns with the traditional definition of resistance to ≥4 antibiotics
    - MAR (Multiple Antibiotic Resistance) Index = resistant_count / antibiotics_tested
    """
    
    # MDR threshold constant - bacteria with MAR > 0.17 are considered MDR
    # This corresponds to resistance to ~4 antibiotics out of 22-23 tested
    MDR_THRESHOLD = 0.17
    
    def __init__(self, models_dir: str = 'outputs/models'):
        """
        Initialize prediction pipeline.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.mar_model = None
        self.species_model = None
        self.feature_cols = None
        self.encoding_map = {'s': 0, 'i': 1, 'r': 2, 'n': 0}  # 'n' (not tested) defaults to susceptible
        self.reverse_encoding = {0: 's', 1: 'i', 2: 'r'}
        
    def load_models(self, mar_model_name: str = 'random_forest',
                    species_model_name: str = 'random_forest'):
        """
        Load trained models.
        
        Args:
            mar_model_name: Name of model for MAR prediction
            species_model_name: Name of model for species prediction
        """
        mar_path = self.models_dir / f'mar_{mar_model_name}.pkl'
        species_path = self.models_dir / f'species_{species_model_name}.pkl'
        
        if mar_path.exists():
            self.mar_model = joblib.load(mar_path)
            print(f"Loaded MAR model from {mar_path}")
        else:
            print(f"Warning: MAR model not found at {mar_path}")
            
        if species_path.exists():
            self.species_model = joblib.load(species_path)
            print(f"Loaded species model from {species_path}")
        else:
            print(f"Warning: Species model not found at {species_path}")
    
    def set_feature_columns(self, feature_cols: List[str]):
        """Set the feature columns used by the models."""
        self.feature_cols = feature_cols
    
    def preprocess_input(self, resistance_profile: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess resistance profile input.
        
        Args:
            resistance_profile: Dictionary mapping antibiotic names to interpretations (s/i/r)
            
        Returns:
            Tuple of (numpy array of encoded features, list of warnings)
        """
        if self.feature_cols is None:
            raise ValueError("Feature columns not set. Call set_feature_columns first.")
        
        features = []
        warnings = []
        
        for col in self.feature_cols:
            # Extract antibiotic name from encoded column name
            antibiotic = col.replace('_encoded', '').replace('_int', '')
            
            # Find matching key in resistance_profile
            value = None
            for key, val in resistance_profile.items():
                if antibiotic in key.lower() or key.lower() in antibiotic:
                    value = val.lower().strip()
                    break
            
            # Encode the value
            if value in self.encoding_map:
                features.append(self.encoding_map[value])
                if value == 'n':
                    # Note that this antibiotic was not tested
                    warnings.append(f"'{antibiotic}' was not tested, defaulting to susceptible (s)")
            elif value is None:
                # Missing antibiotic - default to susceptible but warn
                features.append(0)
                warnings.append(f"Missing value for '{antibiotic}', defaulting to susceptible (s)")
            else:
                # Invalid value - default to susceptible but warn
                features.append(0)
                warnings.append(f"Invalid value '{value}' for '{antibiotic}', defaulting to susceptible (s). "
                              f"Valid values are: s (susceptible), i (intermediate), r (resistant), n (not tested)")
        
        return np.array(features).reshape(1, -1), warnings
    
    def calculate_mar_index(self, resistance_profile: Dict[str, str]) -> Tuple[float, int, int]:
        """
        Calculate MAR (Multiple Antibiotic Resistance) Index from resistance profile.
        
        MAR Index = number of resistant antibiotics / total antibiotics tested
        
        Args:
            resistance_profile: Dictionary mapping antibiotic names to interpretations
            
        Returns:
            Tuple of (mar_index, resistant_count, total_tested)
        """
        resistant_count = 0
        total_tested = 0
        
        for antibiotic, value in resistance_profile.items():
            value_lower = value.lower().strip()
            if value_lower in ['s', 'i', 'r']:  # Only count tested antibiotics
                total_tested += 1
                if value_lower == 'r':
                    resistant_count += 1
        
        mar_index = resistant_count / total_tested if total_tested > 0 else 0.0
        return mar_index, resistant_count, total_tested
    
    def predict_mdr(self, resistance_profile: Dict[str, str]) -> Dict:
        """
        Predict MDR status from resistance profile.
        
        MDR (Multi-Drug Resistance) is determined using MAR index > 0.17 threshold.
        A bacterium is considered MDR if it is resistant to approximately 4 or more
        different antibiotics out of 22-23 typically tested.
        
        Args:
            resistance_profile: Dictionary mapping antibiotic names to interpretations
            
        Returns:
            Dictionary with prediction results including:
            - mdr_prediction: Human-readable MDR status
            - mdr_class: Binary class (1 = MDR, 0 = Non-MDR)
            - mar_index: Calculated MAR index
            - resistant_count: Number of resistant antibiotics
            - mdr_threshold: The threshold used (0.17)
        """
        if self.mar_model is None:
            raise ValueError("MAR model not loaded. Call load_models first.")
        
        X, warnings = self.preprocess_input(resistance_profile)
        
        prediction = self.mar_model.predict(X)[0]
        
        # Calculate actual MAR index for transparency
        mar_index, resistant_count, total_tested = self.calculate_mar_index(resistance_profile)
        
        result = {
            'mdr_prediction': 'MDR (Multi-Drug Resistant)' if prediction == 1 else 'Non-MDR',
            'mdr_class': int(prediction),
            'mar_index': round(mar_index, 4),
            'resistant_count': resistant_count,
            'antibiotics_tested': total_tested,
            'mdr_threshold': self.MDR_THRESHOLD,
            'mdr_explanation': f"MAR Index {mar_index:.4f} {'>' if mar_index > self.MDR_THRESHOLD else '<='} {self.MDR_THRESHOLD} threshold"
        }
        
        # Include warnings if any invalid values were encountered
        if warnings:
            result['warnings'] = warnings
        
        # Get probability if available
        if hasattr(self.mar_model, 'predict_proba'):
            proba = self.mar_model.predict_proba(X)[0]
            result['confidence'] = float(max(proba))
            result['probabilities'] = {
                'Non-MDR': float(proba[0]),
                'MDR': float(proba[1])
            }
        
        return result
    
    def predict_species(self, resistance_profile: Dict[str, str]) -> Dict:
        """
        Predict bacterial species from resistance profile.
        
        Args:
            resistance_profile: Dictionary mapping antibiotic names to interpretations
            
        Returns:
            Dictionary with prediction results
        """
        if self.species_model is None:
            raise ValueError("Species model not loaded. Call load_models first.")
        
        X, warnings = self.preprocess_input(resistance_profile)
        
        prediction = self.species_model.predict(X)[0]
        
        result = {
            'species_prediction': str(prediction)
        }
        
        # Include warnings if any invalid values were encountered
        if warnings:
            result['warnings'] = warnings
        
        # Get probability if available
        if hasattr(self.species_model, 'predict_proba'):
            proba = self.species_model.predict_proba(X)[0]
            result['confidence'] = float(max(proba))
            
            # Get top 3 species predictions
            classes = self.species_model.classes_
            top_indices = np.argsort(proba)[-3:][::-1]
            result['top_predictions'] = [
                {'species': str(classes[i]), 'probability': float(proba[i])}
                for i in top_indices
            ]
        
        return result
    
    def predict_all(self, resistance_profile: Dict[str, str]) -> Dict:
        """
        Run all predictions for a resistance profile.
        
        Args:
            resistance_profile: Dictionary mapping antibiotic names to interpretations
            
        Returns:
            Dictionary with all prediction results
        """
        results = {'input': resistance_profile}
        
        if self.mar_model is not None:
            results['mdr'] = self.predict_mdr(resistance_profile)
        
        if self.species_model is not None:
            results['species'] = self.predict_species(resistance_profile)
        
        return results


# Streamlit Application
STREAMLIT_APP_CODE = '''
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

# Define antibiotics for display (order must match the feature columns used during training)
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

# MDR threshold - bacteria with MAR > 0.17 are considered multi-drug resistant
# This corresponds to resistance to ~4 antibiotics out of 22-23 tested
MDR_THRESHOLD = 0.17

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
    
    try:
        if mar_path.exists():
            mar_model = joblib.load(mar_path)
    except Exception as e:
        st.warning(f"Failed to load MAR model: {e}")
    
    try:
        if species_path.exists():
            species_model = joblib.load(species_path)
    except Exception as e:
        st.warning(f"Failed to load species model: {e}")
    
    return mar_model, species_model


def preprocess_input(resistance_profile):
    """Convert resistance profile to feature array for model prediction."""
    features = []
    for antibiotic in ANTIBIOTICS:
        value = resistance_profile.get(antibiotic, 's').lower()
        features.append(ENCODING_MAP.get(value, 0))
    return np.array(features).reshape(1, -1)


def calculate_mar_index(resistance_profile):
    """
    Calculate MAR (Multiple Antibiotic Resistance) Index.
    
    MAR Index = resistant_count / antibiotics_tested
    MDR is defined as MAR > 0.17 (resistance to ~4+ antibiotics)
    """
    resistant_count = 0
    tested_count = 0
    
    for antibiotic in ANTIBIOTICS:
        value = resistance_profile.get(antibiotic, 'n').lower()
        if value in ['s', 'i', 'r']:  # Only count tested antibiotics
            tested_count += 1
            if value == 'r':
                resistant_count += 1
    
    mar_index = resistant_count / tested_count if tested_count > 0 else 0.0
    return mar_index, resistant_count, tested_count


# Title and description
st.title("🦠 AMR Pattern Recognition System")
st.markdown("""
This application predicts:
- **MDR (Multi-Drug Resistance)** status based on antibiotic resistance profiles
- **Bacterial Species** from resistance patterns

**MDR Definition:** A bacterium is considered multi-drug resistant (MDR) if its MAR index exceeds 0.17,
which corresponds to resistance to approximately 4 or more antibiotics.

Upload your data or enter antibiotic resistance values manually.
""")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown(f"**MDR Threshold:** MAR > {MDR_THRESHOLD}")
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
        
        # Calculate MAR index
        mar_index, resistant_count, tested_count = calculate_mar_index(resistance_profile)
        
        # Display MAR index calculation
        st.markdown("---")
        st.markdown("### MAR Index Analysis")
        mar_col1, mar_col2, mar_col3 = st.columns(3)
        with mar_col1:
            st.metric("Resistant Antibiotics", resistant_count)
        with mar_col2:
            st.metric("Antibiotics Tested", tested_count)
        with mar_col3:
            st.metric("MAR Index", f"{mar_index:.4f}")
        
        is_mdr_by_mar = mar_index > MDR_THRESHOLD
        if is_mdr_by_mar:
            st.warning(f"⚠️ MAR Index ({mar_index:.4f}) > {MDR_THRESHOLD} threshold → **MDR**")
        else:
            st.success(f"✅ MAR Index ({mar_index:.4f}) ≤ {MDR_THRESHOLD} threshold → **Non-MDR**")
        
        st.markdown("---")
        st.markdown("### Model Predictions")
        col1, col2 = st.columns(2)
        
        with col1:
            if mar_model is not None:
                mar_pred = mar_model.predict(X)[0]
                mar_label = "MDR (Multi-Drug Resistant)" if mar_pred == 1 else "Non-MDR"
                
                if hasattr(mar_model, 'predict_proba'):
                    mar_proba = mar_model.predict_proba(X)[0]
                    mar_confidence = max(mar_proba) * 100
                    st.metric(
                        label="MDR Prediction (Model)",
                        value=mar_label,
                        delta=f"{mar_confidence:.1f}% confidence"
                    )
                else:
                    st.metric(
                        label="MDR Prediction (Model)",
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
                    
                    # Calculate MAR index
                    mar_index, resistant_count, tested_count = calculate_mar_index(profile)
                    result['mar_index'] = round(mar_index, 4)
                    result['resistant_count'] = resistant_count
                    
                    if mar_model is not None:
                        mar_pred = mar_model.predict(X)[0]
                        result['mdr_prediction'] = "MDR" if mar_pred == 1 else "Non-MDR"
                    
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
        'Antibiotic': [a.replace("_", " ").replace("/", " / ").title() for a in ANTIBIOTICS[:10]],
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
'''


# FastAPI Application
FASTAPI_APP_CODE = '''
"""
AMR Pattern Recognition - FastAPI REST API

Run with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="AMR Pattern Recognition API",
    description="API for predicting MDR status and bacterial species from antibiotic resistance profiles",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models (load at startup)
MODELS_DIR = Path("outputs/models")
mar_model = None
species_model = None

# Encoding map
ENCODING_MAP = {'s': 0, 'i': 1, 'r': 2, 'n': 0}  # 'n' (not tested) defaults to susceptible

# MDR threshold - bacteria with MAR > 0.17 are considered multi-drug resistant
# This corresponds to resistance to ~4 antibiotics out of 22-23 tested
MDR_THRESHOLD = 0.17


def calculate_mar_index(antibiotics: Dict[str, str]):
    """
    Calculate MAR (Multiple Antibiotic Resistance) Index.
    
    MAR Index = resistant_count / antibiotics_tested
    MDR is defined as MAR > 0.17 (resistance to ~4+ antibiotics)
    """
    resistant_count = 0
    tested_count = 0
    
    for antibiotic, value in antibiotics.items():
        value_lower = value.lower().strip()
        if value_lower in ['s', 'i', 'r']:  # Only count tested antibiotics
            tested_count += 1
            if value_lower == 'r':
                resistant_count += 1
    
    mar_index = resistant_count / tested_count if tested_count > 0 else 0.0
    return mar_index, resistant_count, tested_count


class ResistanceProfile(BaseModel):
    """Input model for resistance profile"""
    antibiotics: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "antibiotics": {
                    "ampicillin": "r",
                    "gentamicin": "s",
                    "tetracycline": "i"
                }
            }
        }


class MDRPrediction(BaseModel):
    """Output model for MDR prediction"""
    mdr_status: str
    mdr_class: int
    mar_index: float
    resistant_count: int
    antibiotics_tested: int
    mdr_threshold: float
    confidence: float
    probabilities: Dict[str, float]


class SpeciesPrediction(BaseModel):
    """Output model for species prediction"""
    predicted_species: str
    confidence: float
    top_predictions: List[Dict[str, float]]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global mar_model, species_model
    
    mar_path = MODELS_DIR / "mar_random_forest.pkl"
    species_path = MODELS_DIR / "species_random_forest.pkl"
    
    if mar_path.exists():
        mar_model = joblib.load(mar_path)
        print(f"Loaded MAR model from {mar_path}")
    
    if species_path.exists():
        species_model = joblib.load(species_path)
        print(f"Loaded species model from {species_path}")


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "mar_model": mar_model is not None,
            "species_model": species_model is not None
        }
    )


@app.post("/predict/mdr", response_model=MDRPrediction)
async def predict_mdr(profile: ResistanceProfile):
    """
    Predict MDR (Multi-Drug Resistance) status from resistance profile.
    
    MDR is determined using MAR index > 0.17 threshold.
    A bacterium is considered MDR if it is resistant to approximately 4 or more
    different antibiotics out of 22-23 typically tested.
    
    Returns:
        MDR prediction with confidence scores and MAR index calculation
    """
    if mar_model is None:
        raise HTTPException(status_code=503, detail="MAR model not loaded")
    
    # Calculate MAR index
    mar_index, resistant_count, tested_count = calculate_mar_index(profile.antibiotics)
    
    # Preprocess input
    features = [ENCODING_MAP.get(v.lower(), 0) for v in profile.antibiotics.values()]
    X = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = mar_model.predict(X)[0]
    proba = mar_model.predict_proba(X)[0] if hasattr(mar_model, 'predict_proba') else [0.5, 0.5]
    
    return MDRPrediction(
        mdr_status="MDR (Multi-Drug Resistant)" if prediction == 1 else "Non-MDR",
        mdr_class=int(prediction),
        mar_index=round(mar_index, 4),
        resistant_count=resistant_count,
        antibiotics_tested=tested_count,
        mdr_threshold=MDR_THRESHOLD,
        confidence=float(max(proba)),
        probabilities={
            "Non-MDR": float(proba[0]),
            "MDR": float(proba[1])
        }
    )


@app.post("/predict/species", response_model=SpeciesPrediction)
async def predict_species(profile: ResistanceProfile):
    """
    Predict bacterial species from resistance profile.
    
    Returns:
        Species prediction with confidence scores
    """
    if species_model is None:
        raise HTTPException(status_code=503, detail="Species model not loaded")
    
    # Preprocess input
    features = [ENCODING_MAP.get(v.lower(), 0) for v in profile.antibiotics.values()]
    X = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = species_model.predict(X)[0]
    proba = species_model.predict_proba(X)[0] if hasattr(species_model, 'predict_proba') else [1.0]
    
    # Get top predictions
    classes = species_model.classes_
    top_indices = np.argsort(proba)[-3:][::-1]
    top_predictions = [
        {"species": str(classes[i]), "probability": float(proba[i])}
        for i in top_indices
    ]
    
    return SpeciesPrediction(
        predicted_species=str(prediction),
        confidence=float(max(proba)),
        top_predictions=top_predictions
    )


@app.post("/predict/all")
async def predict_all(profile: ResistanceProfile):
    """
    Run all predictions for a resistance profile.
    
    Returns:
        Combined MDR and species predictions
    """
    results = {"input": profile.antibiotics}
    
    if mar_model is not None:
        mdr_result = await predict_mdr(profile)
        results["mdr"] = mdr_result.dict()
    
    if species_model is not None:
        species_result = await predict_species(profile)
        results["species"] = species_result.dict()
    
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''


def create_deployment_files(output_dir: str = '.'):
    """
    Create deployment files (Streamlit app and FastAPI).
    
    Args:
        output_dir: Directory to save files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Streamlit app
    with open(f'{output_dir}/app.py', 'w') as f:
        f.write(STREAMLIT_APP_CODE)
    print(f"Created Streamlit app at {output_dir}/app.py")
    
    # Create FastAPI app
    with open(f'{output_dir}/api.py', 'w') as f:
        f.write(FASTAPI_APP_CODE)
    print(f"Created FastAPI app at {output_dir}/api.py")


def run_phase4(output_dir: str = '.') -> Dict:
    """
    Run complete Phase 4: Deployment Preparation
    
    Args:
        output_dir: Directory to save deployment files
        
    Returns:
        Dictionary with deployment info
    """
    print("=" * 60)
    print("PHASE 4: DEPLOYMENT PREPARATION")
    print("=" * 60)
    
    # Create deployment files
    create_deployment_files(output_dir)
    
    print("\n" + "-" * 40)
    print("Deployment Files Created")
    print("-" * 40)
    print(f"""
Files created:
- {output_dir}/app.py (Streamlit web application)
- {output_dir}/api.py (FastAPI REST API)

To run:
1. Streamlit: streamlit run app.py
2. FastAPI: uvicorn api:app --reload

Deployment options:
- HuggingFace Spaces (recommended for Streamlit)
- Heroku
- AWS/GCP/Azure
- Docker container
    """)
    
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    
    return {
        'streamlit_app': f'{output_dir}/app.py',
        'fastapi_app': f'{output_dir}/api.py'
    }


if __name__ == "__main__":
    run_phase4('.')
