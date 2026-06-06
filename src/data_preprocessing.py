import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_and_feature_engineering(file_path):
    """
    Advanced preprocessing pipeline: Fixes dates, standardizes property sizes,
    applies target encoding to locations, one-hot encodes categories, and scales features.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find raw file at: {file_path}")

    print("🔄 Step 1: Loading raw dataset...")
    df = pd.read_csv(file_path)
    print(f"   Initial Shape: {df.shape}")

    # --- 1. FIXED DATE_ADDED CLEANING ---
    print("\n📅 Step 2: Repairing and formatting 'date_added' column...")
    date_col = next((c for c in ["date_added", "dateadded", "Date Added"] if c in df.columns), None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isnull().sum() > 0:
            df[date_col] = df[date_col].fillna(df[date_col].mode()[0])
        df['date_added_clean'] = df[date_col].dt.strftime('%Y-%m-%d')
        df = df.drop(columns=[date_col])
        print("   -> Successfully standardized dates to YYYY-MM-DD format.")
    else:
        print("   ⚠️ No date tracking column detected.")

    # --- 2. DATA FILTERING & CLEANING ---
    print("\n⚖️ Step 3: Filtering properties and balancing target classes...")
    if "purpose" in df.columns:
        df = df[df["purpose"] == "For Sale"]
        df = df.drop(columns=["purpose"])

    if "property_type" in df.columns:
        valid_types = ["House", "Flat", "Upper Portion", "Lower Portion"]
        df = df[df["property_type"].isin(valid_types)]
        print(f"   -> Retained primary residential types: {valid_types}")

    # --- 3. MATHEMATICAL AREA STANDARDIZATION ---
    size_col = "Area Size" if "Area Size" in df.columns else "area"
    type_col = "Area Type" if "Area Type" in df.columns else "area_type"

    if type_col in df.columns and size_col in df.columns:
        print("   -> Mathematically converting all property sizes into Square Feet...")
        df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
        df = df.dropna(subset=[size_col])

        df["area_sqft"] = df.apply(
            lambda row: row[size_col] * 4500 if row[type_col] == "Kanal" else row[size_col] * 225,
            axis=1
        )
        df = df.drop(columns=["area", "Area Size", "Area Type", "Area Category"], errors="ignore")

    df = df.drop(columns=["page_url", "agency", "agent", "property_id"], errors="ignore")

    # Dynamic Column Mapping for structural configurations
    bed_col = next((c for c in ["bedrooms", "beds", "Bedrooms"] if c in df.columns), None)
    bath_col = next((c for c in ["bathrooms", "baths", "Baths"] if c in df.columns), None)

    if bed_col and bath_col:
        df = df.rename(columns={bed_col: "bedrooms", bath_col: "bathrooms"})
        df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
        df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    print(f"   Shape after filtering/balancing steps: {df.shape}")

    # --- 4. TARGET VARIABLE TRANSFORMATION (LOG TRANSFORM) ---
    print("\n📈 Step 4: Stabilizing variance by log-transforming target 'price'...")
    df["log_price"] = np.log1p(df["price"])

    y = df["log_price"]
    X = df.drop(columns=["price", "log_price"])

    # --- 5. SYSTEM DATA SPLITTING (TRAIN/TEST) ---
    print("\n✂️ Step 5: Partitioning data into Train and Test matrices to prepare for Target Encoding...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 6. TARGET ENCODING ON HIGH-CARDINALITY 'LOCATION' ---
    if "location" in X_train.columns:
        print("\n🗺️ Step 6: Implementing structural Target Encoding on 'location' column...")
        train_df = X_train.copy()
        train_df['target_log_price'] = y_train
        location_means = train_df.groupby('location')['target_log_price'].mean()

        global_mean = y_train.mean()
        X_train['location_encoded'] = X_train['location'].map(location_means).fillna(global_mean)
        X_test['location_encoded'] = X_test['location'].map(location_means).fillna(global_mean)

        X_train = X_train.drop(columns=['location'])
        X_test = X_test.drop(columns=['location'])
        print("   -> Successfully encoded location strings to average log-target scalar footprints.")

    # --- 7. ONE-HOT ENCODING ON CATEGORICAL COLUMNS ---
    print("\n🔠 Step 7: Executing One-Hot Encoding on nominal categories...")
    categorical_features = ["property_type", "city", "province_name"]
    categorical_features = [col for col in categorical_features if col in X_train.columns]

    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    print(f"   Features dimension after encoding expansion: {X_train_encoded.shape}")

    # --- 8. FEATURE SCALING USING STANDARD SCALER ---
    print("\n📐 Step 8: Standardizing continuous features via StandardScaler...")
    continuous_cols = ["bedrooms", "bathrooms", "area_sqft"]
    spatial_cols = [c for c in ["latitude", "longitude"] if c in X_train_encoded.columns]
    scale_targets = continuous_cols + spatial_cols

    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()

    X_train_scaled[scale_targets] = scaler.fit_transform(X_train_encoded[scale_targets])
    X_test_scaled[scale_targets] = scaler.transform(X_test_encoded[scale_targets])

    print("✅ Full preprocessing pipeline finished successfully without data leakage.")
    return X_train_scaled, X_test_scaled, y_train, y_test


# --- Execution Block ---
if __name__ == "__main__":
    RAW_FILE_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Raw\zameen-updated.csv"

    # Target output paths for Excel/CSV viewing
    PROCESSED_DIR = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed"
    TRAIN_EXPORT_PATH = os.path.join(PROCESSED_DIR, "zameen_train_preprocessed.csv")
    TEST_EXPORT_PATH = os.path.join(PROCESSED_DIR, "zameen_test_preprocessed.csv")

    # Process matrices directly
    X_train, X_test, y_train, y_test = preprocess_and_feature_engineering(RAW_FILE_PATH)

    # --- 9. EXPORTING PREPROCESSED DATA SHEETS ---
    print("\n💾 Step 9: Exporting fully preprocessed splits into Excel-readable files...")

    # Ensure processed storage directory existsgi
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Merge targets back into dataframes so rows match up side-by-side
    final_train_df = X_train.copy()
    final_train_df["target_log_price"] = y_train

    final_test_df = X_test.copy()
    final_test_df["target_log_price"] = y_test

    # Save files out
    final_train_df.to_csv(TRAIN_EXPORT_PATH, index=False)
    final_test_df.to_csv(TEST_EXPORT_PATH, index=False)

    print(f"   📥 Training Preprocessed Sheet exported to: {TRAIN_EXPORT_PATH}")
    print(f"   📥 Testing Preprocessed Sheet exported to: {TEST_EXPORT_PATH}")

    print("\n" + "=" * 50)
    print("🚀 PIPELINE SUCCESS PREVIEW")
    print("=" * 50)
    print(f"Exported Training Rows: {final_train_df.shape[0]} | Columns: {final_train_df.shape[1]}")
    print(f"Exported Testing Rows: {final_test_df.shape[0]} | Columns: {final_test_df.shape[1]}")