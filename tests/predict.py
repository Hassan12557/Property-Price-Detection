import os
import joblib
import numpy as np
import pandas as pd


def predict_single_property(query_data, models_dir):
    """Loads operational preprocessing assets and your champion model checkpoint to

    compute a localized property price inference prediction.
    """
    # 1. Verify existence of production weights and maps
    pipeline_asset_path = os.path.join(models_dir, "data_pipeline_assets.joblib")
    model_checkpoint_path = os.path.join(models_dir, "model.pkl")

    if not os.path.exists(pipeline_asset_path) or not os.path.exists(
        model_checkpoint_path
    ):
        raise FileNotFoundError(
            "Production assets missing in models/ directory. "
            "Please ensure both data_pipeline.py and train_models.py have run successfully."
        )

    # 2. De-serialize structural intelligence assets
    print("📦 Loading serialization parameters and model engine weights...")
    pipeline_assets = joblib.load(pipeline_asset_path)
    model = joblib.load(model_checkpoint_path)

    scaler = pipeline_assets["scaler"]
    location_map = pipeline_assets["location_map"]
    global_mean = pipeline_assets["global_mean"]
    feature_columns = pipeline_assets["feature_columns"]

    # 3. Restructure incoming test data dictionary into a DataFrame row
    input_df = pd.DataFrame([query_data])

    # 4. STEP 6 REPLICATOR: Safe Target Encoding mapping logic
    if "location" in input_df.columns:
        loc_val = input_df.loc[0, "location"]
        # Look up location in training map. If it's a completely new location, fallback safely to global mean
        input_df["location_encoded"] = location_map.get(loc_val, global_mean)
        input_df = input_df.drop(columns=["location"])

    # 5. STEP 7 REPLICATOR: One-Hot Categorical Expansion Alignment
    # Generate dummies for current prediction row
    cat_cols = ["property_type", "city", "province_name"]
    input_df_enc = pd.get_dummies(input_df, columns=cat_cols, drop_first=False)

    # Re-index columns matching the original training frame, filling missing expansions with 0 (False)
    input_df_final = input_df_enc.reindex(columns=feature_columns, fill_value=0)

    # 6. STEP 8 REPLICATOR: StandardScaler Normalization
    # Scale mathematical variables using fitted parameters from training data
    scale_targets = ["bedrooms", "bathroom", "area_sqft"]
    spatial_targets = [
        c for c in ["latitude", "longitude"] if c in input_df_final.columns
    ]
    all_scale_cols = scale_targets + spatial_targets

    input_df_final[all_scale_cols] = scaler.transform(
        input_df_final[all_scale_cols]
    )

    # 7. EXECUTE INFERENCE PREDICTION
    print("🔮 Running mathematical model inference forward pass...")
    log_prediction = model.predict(input_df_final)[0]

    # 8. REVERSE TARGET LOG MATHEMATICS (Convert log_price back to raw PKR)
    raw_pkr_price = np.expm1(log_prediction)

    return raw_pkr_price


if __name__ == "__main__":
    # Path to your unified serialized models directory folder
    MODELS_DIRECTORY = (
        r"D:\Data Science Projects\Property-Price-Detection\models"
    )

    # Define a completely custom mock property input profile dictionary to test
    # (Input raw values exactly as a real user would type them into an app)
    test_property_profile = {
        "bedrooms": 3,
        "bathroom": 3,
        "area_sqft": 1125.0,  # equivalent to a standard 5 Marla house profile
        "location": "DHA Defence",  # Target encoding will handle this string
        "latitude": 33.5351,  # Sample coordinates
        "longitude": 73.1324,
        "property_type": "House",
        "city": "Islamabad",
        "province_name": "Islamabad Capital Territory",
    }

    print("=" * 65)
    print("🏢 REAL ESTATE PROPERTY INFERENCE EXPERIMENTAL SIMULATOR")
    print("=" * 65)
    print("Target Test Profile:")
    for key, val in test_property_profile.items():
        print(f"  🔹 {key}: {val}")
    print("-" * 65)

    try:
        predicted_market_value = predict_single_property(
            test_property_profile, MODELS_DIRECTORY
        )

        # Print beautifully formatted currency calculations
        print("\n" + "🚀" * 15)
        print(f"💰 EVALUATED ESTIMATED PROPERTY VALUE:")
        print(f"   PKR {predicted_market_value:,.2f}")
        print(f"   (~ {predicted_market_value / 1000000:.2f} Million PKR)")
        print("🚀" * 15)

    except Exception as e:
        print(f"\n❌ Prediction Pipeline Halted: {str(e)}")
