
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
