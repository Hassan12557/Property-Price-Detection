import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_and_feature_engineering(file_path):
    """Loads, cleans, extracts numerical sizes, filters imbalances, encodes,

    and scales properties for machine learning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find raw file at: {file_path}")

    print("🔄 Step 1: Loading raw dataset...")
    df = pd.read_csv(file_path)
    print(f"   Initial Shape: {df.shape}")

    # --- 1. HANDLING DATA BALANCING & FILTERING ---
    print("\n⚖️ Step 2: Balancing datasets through targeted filtering...")

    # Filter Purpose: Isolate 'For Sale' rows to prevent target contamination
    if "purpose" in df.columns:
        df = df[df["purpose"] == "For Sale"]
        df = df.drop(columns=["purpose"])
        print("   -> Dropped rental properties; filtered strictly for 'For Sale'.")

    # Filter Property Type: Retain major classes, drop rare outliers (< 1% representation)
    if "property_type" in df.columns:
        valid_types = ["House", "Flat", "Upper Portion", "Lower Portion"]
        df = df[df["property_type"].isin(valid_types)]
        print(f"   -> Dropped minor property types. Retained: {valid_types}")

    # CLEAN AND EXTRACT NUMERIC AREA (Fixes the '5.5 Marla' string replication bug)
    if "Area Type" in df.columns and "area" in df.columns:
        print(
            "   -> Extracting numbers from text area entries (e.g., '5.5 Marla' -> 5.5)..."
        )

        # Force column to string, extract decimals/integers, and convert to float
        df["area_cleaned"] = (
            df["area"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]
            .astype(float)
        )

        # Drop rows where area extraction failed or resulted in 0
        df = df.dropna(subset=["area_cleaned"])
        df = df[df["area_cleaned"] > 0]

        print(
            "   -> Transforming structural 'Area Type' categories into numerical Square Feet..."
        )
        # Apply conversion math securely on clean floats
        df["area_sqft"] = df.apply(
            lambda row: (
                row["area_cleaned"] * 4500
                if row["Area Type"] == "Kanal"
                else row["area_cleaned"] * 225
            ),
            axis=1,
        )

        # Drop raw text mapping helpers now that sizes are balanced and continuous
        df = df.drop(
            columns=["area", "area_cleaned", "Area Type", "Area Category"],
            errors="ignore",
        )

    # Drop non-predictive variables to prevent index cluttering
    df = df.drop(
        columns=["page_url", "date_added", "location"], errors="ignore"
    )

    # --- DYNAMIC COLUMN NAME RESOLUTION FOR BEDS & BATHS ---
    bed_col = None
    for alt in ["bedrooms", "beds", "Bedrooms", "Beds", "beds_count"]:
        if alt in df.columns:
            bed_col = alt
            break

    bath_col = None
    for alt in ["bathrooms", "baths", "Bathrooms", "Baths", "baths_count"]:
        if alt in df.columns:
            bath_col = alt
            break

    if not bed_col or not bath_col:
        print(
            "\n❌ CRITICAL ERROR: Could not map bathroom or bedroom columns automatically."
        )
        print(f"Available columns in your dataset are: {list(df.columns)}")
        raise KeyError(
            "Missing required structural columns. Match column names exactly."
        )

    # Standardize column headers into a unified clean name format
    if bed_col != "bedrooms":
        df = df.rename(columns={bed_col: "bedrooms"})
    if bath_col != "bathrooms":
        df = df.rename(columns={bath_col: "bathrooms"})

    print(
        f"   -> Successfully matched structural columns: '{bed_col}' ➔ 'bedrooms', '{bath_col}' ➔ 'bathrooms'"
    )

    # Handle missing value fallbacks safely using our resolved uniform names
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
    df = df.dropna(subset=["price"])  # Eliminate target holes

    # Safely remove any rows with zero or negative prices before applying log1p
    df = df[df["price"] > 0]

    print(f"   Shape after filtering/balancing: {df.shape}")

    # --- 2. TARGET TRANSFORMATION & COV SCALE ---
    print("\n📈 Step 3: Handling price skewness transformation...")
    df["log_price"] = np.log1p(df["price"])

    # Isolate targets (y) from features (X)
    y = df["log_price"]
    X = df.drop(columns=["price", "log_price"])

    # --- 3. ONE-HOT ENCODING ---
    print("\n🔠 Step 4: Applying One-Hot Encoding on categorical attributes...")
    categorical_features = ["property_type", "city", "province_name"]
    categorical_features = [col for col in categorical_features if col in X.columns]

    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    print(f"   Features shape after encoding expansion: {X_encoded.shape}")

    # --- 4. DATASET SPLITTING ---
    print("\n✂️ Step 5: Partitioning datasets into Train and Test matrices...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    print(f"   Training Dimensions: {X_train.shape}")
    print(f"   Testing Dimensions: {X_test.shape}")

    # --- 5. FEATURE SCALING ---
    print("\n📐 Step 6: Engineering scaling features using StandardScaler...")
    continuous_cols = ["bedrooms", "bathrooms", "area_sqft"]

    scaler = StandardScaler()

    # Fit on training data and transform both to avoid information data leakage
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_cols] = scaler.fit_transform(
        X_train[continuous_cols]
    )
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])

    print("✅ Full preprocessing pipeline finished successfully.")
    return X_train_scaled, X_test_scaled, y_train, y_test


# --- Execution Block ---
if __name__ == "__main__":
    RAW_FILE_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Raw\zameen-updated.csv"

    # Process matrices directly
    X_train, X_test, y_train, y_test = preprocess_and_feature_engineering(
        RAW_FILE_PATH
    )

    # Sanity verification print
    print("\n" + "=" * 50)
    print("🚀 PIPELINE SUCCESS PREVIEW")
    print("=" * 50)
    print("First 3 processed rows of Training Features (X_train):")
    print(X_train.head(3))