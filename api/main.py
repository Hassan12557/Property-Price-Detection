import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- DIRECTORY CONFIGURATIONS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- INITIALIZE API & LOAD ASSETS ---
app = FastAPI(
    title="Zameen Real Estate Valuation API",
    description="Production-ready REST API delivering real-time machine learning inference for property prices in Pakistan.",
    version="1.0.0"
)

def load_ml_components():
    pipeline_path = os.path.join(MODELS_DIR, "data_pipeline_assets.joblib")
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    
    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        return None, None
    return joblib.load(pipeline_path), joblib.load(model_path)

pipeline_assets, model = load_ml_components()

if pipeline_assets is None or model is None:
    raise RuntimeError("System Core Failure: Serialization model assets missing from disk storage arrays.")

# Extract necessary preprocessing parameters
feature_columns = pipeline_assets["feature_columns"]
location_map = pipeline_assets["location_map"]
global_mean = pipeline_assets["global_mean"]
scaler = pipeline_assets["scaler"]


# --- PYDANTIC SCHEMAS (Reduces Cognitive Load with Strict Data Validation) ---
class PropertyInferenceRequest(BaseModel):
    bedrooms: int = Field(..., ge=1, le=12, description="Total number of bedrooms", example=3)
    bathroom: int = Field(..., ge=1, le=12, description="Total number of bathrooms", example=3)
    area_size: float = Field(..., ge=0.5, description="Size number matching the unit choice", example=5.0)
    area_type: str = Field(..., description="Measurement unit scale chosen", example="Marla")
    property_type: str = Field(..., description="Property class design classification", example="House")
    city: str = Field(..., description="Metropolitan targeted urban city location", example="Islamabad")
    location: str = Field(..., description="Target sector or development colony neighborhood", example="G-10")
    province_name: str = Field(..., description="Macro provincial boundary regional name", example="Islamabad Capital Territory")
    latitude: float = Field(default=33.6844, description="Optional high-fidelity geographical mapping marker")
    longitude: float = Field(default=73.0479, description="Optional high-fidelity geographical mapping marker")

class PricePredictionResponse(BaseModel):
    status: str = Field(..., example="success")
    predicted_price_pkr: float = Field(..., description="Calculated property valuation in absolute PKR values", example=14500000.0)
    readable_valuation: str = Field(..., description="Human legible currency string break out breakdown", example="1.45 Crore PKR")


# --- API ENDPOINTS ---

@app.get("/", tags=["System Diagnostics"])
def read_root():
    """ Health-check system entry confirmation lane. """
    return {
        "status": "online",
        "model_loaded": model is not None,
        "active_features_count": len(feature_columns)
    }

@app.post("/predict", response_model=PricePredictionResponse, tags=["Machine Learning Inference"])
def predict_property_price(payload: PropertyInferenceRequest):
    """
    Accepts a structured JSON profile packet, applies pipeline conversions, normalizes values, 
    and computes log-space real estate valuations returned as final currency estimations.
    """
    try:
        # 1. Coordinate spatial footprint transformations behind the scenes
        converted_sqft = payload.area_size * 4500 if payload.area_type.strip().lower() == "kanal" else payload.area_size * 225
        
        # 2. Re-map data request array into processing structure
        query_dict = {
            "bedrooms": float(payload.bedrooms),
            "bathroom": float(payload.bathroom),
            "area_sqft": float(converted_sqft),
            "latitude": float(payload.latitude),
            "longitude": float(payload.longitude),
            "property_type": payload.property_type,
            "city": payload.city,
            "province_name": payload.province_name
        }
        
        eval_df = pd.DataFrame([query_dict])
        
        # 3. Inject serialized target maps
        eval_df["location_encoded"] = location_map.get(payload.location, global_mean)
        
        # 4. Generate one-hot encoded matrix tracks matching original training arrays
        eval_df_enc = pd.get_dummies(eval_df, columns=["property_type", "city", "province_name"])
        eval_df_final = eval_df_enc.reindex(columns=feature_columns, fill_value=0)
        
        # 5. Apply numerical distribution min-max tracking scaling variables
        scale_cols = ["bedrooms", "bathroom", "area_sqft", "latitude", "longitude"]
        scale_cols = [c for c in scale_cols if c in eval_df_final.columns]
        eval_df_final[scale_cols] = scaler.transform(eval_df_final[scale_cols])
        
        # 6. Execute predictive inference forward calculation
        predicted_log_price = model.predict(eval_df_final)[0]
        final_price = np.expm1(predicted_log_price)
        
        # Formulate readable text outputs
        readable_str = f"{final_price / 10000000:.2f} Crore PKR" if final_price >= 10000000 else f"{final_price / 100000:.2f} Lakh PKR"

        return PricePredictionResponse(
            status="success",
            predicted_price_pkr=round(final_price, 2),
            readable_valuation=readable_str
        )
        
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failure: {str(err)}")
