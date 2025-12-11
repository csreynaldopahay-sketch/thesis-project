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
    - MDR/MAR prediction
    - Species prediction
    - Feature extraction
    """
    
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
        self.encoding_map = {'s': 0, 'i': 1, 'r': 2}
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
            elif value is None:
                # Missing antibiotic - default to susceptible but warn
                features.append(0)
                warnings.append(f"Missing value for '{antibiotic}', defaulting to susceptible (s)")
            else:
                # Invalid value - default to susceptible but warn
                features.append(0)
                warnings.append(f"Invalid value '{value}' for '{antibiotic}', defaulting to susceptible (s). "
                              f"Valid values are: s (susceptible), i (intermediate), r (resistant)")
        
        return np.array(features).reshape(1, -1), warnings
    
    def predict_mdr(self, resistance_profile: Dict[str, str]) -> Dict:
        """
        Predict MDR status from resistance profile.
        
        Args:
            resistance_profile: Dictionary mapping antibiotic names to interpretations
            
        Returns:
            Dictionary with prediction results
        """
        if self.mar_model is None:
            raise ValueError("MAR model not loaded. Call load_models first.")
        
        X, warnings = self.preprocess_input(resistance_profile)
        
        prediction = self.mar_model.predict(X)[0]
        
        result = {
            'mdr_prediction': 'High MAR (MDR)' if prediction == 1 else 'Low MAR',
            'mdr_class': int(prediction)
        }
        
        # Include warnings if any invalid values were encountered
        if warnings:
            result['warnings'] = warnings
        
        # Get probability if available
        if hasattr(self.mar_model, 'predict_proba'):
            proba = self.mar_model.predict_proba(X)[0]
            result['confidence'] = float(max(proba))
            result['probabilities'] = {
                'Low MAR': float(proba[0]),
                'High MAR (MDR)': float(proba[1])
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
ENCODING_MAP = {'s': 0, 'i': 1, 'r': 2}


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
    Predict MDR status from resistance profile.
    
    Returns:
        MDR prediction with confidence scores
    """
    if mar_model is None:
        raise HTTPException(status_code=503, detail="MAR model not loaded")
    
    # Preprocess input
    features = [ENCODING_MAP.get(v.lower(), 0) for v in profile.antibiotics.values()]
    X = np.array(features).reshape(1, -1)
    
    # Predict
    prediction = mar_model.predict(X)[0]
    proba = mar_model.predict_proba(X)[0] if hasattr(mar_model, 'predict_proba') else [0.5, 0.5]
    
    return MDRPrediction(
        mdr_status="High MAR (MDR)" if prediction == 1 else "Low MAR",
        mdr_class=int(prediction),
        confidence=float(max(proba)),
        probabilities={
            "Low MAR": float(proba[0]),
            "High MAR (MDR)": float(proba[1])
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
