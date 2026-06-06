import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ZameenDataPipeline:
    def __init__(self, raw_data_path, processed_dir, artifacts_dir):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        self.artifacts_dir = artifacts_dir
        
        # Initialize pipeline assets
        self.scaler = StandardScaler()
        self.location_encoder_map = {}
        self.global_log_price_mean = 0.0
        self.training_feature_columns = []

        # Core continuous features to scale
        self.continuous_cols = ["bedrooms", "bathroom", "area_sqft"]
        # Include spatial coordinates if they successfully persist in data
        self.spatial_cols = ["latitude", "longitude"]

    def load_raw_data(self):
        """Loads dataset from local physical drive path."""
        print("🔄 Step 1: Loading raw Zameen dataset...")
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Missing raw CSV matrix target at: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        print(f"   📥 Initial Shape: {df.shape}")
        return df

    def clean_and_filter(self, df):
        """Standardizes date types, drops unneeded rows, and eliminates class imbalances."""
        print("\n🧹 Step 2: Running target cleaning and filtering parameters...")

        # 1. Date formatting and missing date imputation
        date_col = next((c for c in ["date_added", "dateadded", "Date Added"] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            if df[date_col].isnull().sum() > 0:
                df[date_col] = df[date_col].fillna(df[date_col].mode()[0])
            df['date_added_clean'] = df[date_col].dt.strftime('%Y-%m-%d')
            df = df.drop(columns=[date_col])

        # 2. Prevent target contamination: Filter solely for active sales listings
        if "purpose" in df.columns:
            df = df[df["purpose"] == "For Sale"].drop(columns=["purpose"])

        # 3. Drop minor property types representing < 1% of market presence
        if "property_type" in df.columns:
            valid_types = ["House", "Flat", "Upper Portion", "Lower Portion"]
            df = df[df["property_type"].isin(valid_types)]

        # 4. Standardize bed and bath columns dynamically
        bed_col = next((c for c in ["bedrooms", "beds", "Bedrooms"] if c in df.columns), None)
        bath_col = next((c for c in ["bathrooms", "baths", "Baths", "bath"] if c in df.columns), None)
        if bed_col and bath_col:
            df = df.rename(columns={bed_col: "bedrooms", bath_col: "bathroom"})
            df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
            df["bathroom"] = df["bathroom"].fillna(df["bathroom"].median())

        # 5. Drop administrative columns that degrade training quality
        df = df.drop(columns=["page_url", "agency", "agent", "property_id"], errors="ignore")

        # 6. Eliminate corrupted row targets
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0]
        
        return df

    def standardize_area_metrics(self, df):
        """Applies real estate math transformations to unify all area measures to Sq Ft."""
        print("\n📐 Step 3: Standardizing property footprint metrics to Square Feet...")
        size_col = "Area Size" if "Area Size" in df.columns else "area"
        type_col = "Area Type" if "Area Type" in df.columns else "area_type"

        if type_col in df.columns and size_col in df.columns:
            df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
            df = df.dropna(subset=[size_col])
            
            # Apply mathematical unit conversions (1 Kanal = 4500 sqft, 1 Marla = 225 sqft)
            df["area_sqft"] = df.apply(
                lambda row: row[size_col] * 4500 if row[type_col] == "Kanal" else row[size_col] * 225,
                axis=1
            )
            df = df.drop(columns=["area", "Area Size", "Area Type", "Area Category"], errors="ignore")
        else:
            raise KeyError("Critical structural columns ('Area Size' / 'Area Type') missing from file dataframe.")
        return df

    def execute_pipeline(self):
        """Runs the data processing pipeline end-to-end."""
        # Load and execute functional transformations
        df_raw = self.load_raw_data()
        df_cleaned = self.clean_and_filter(df_raw)
        df_standardized = self.standardize_area_metrics(df_cleaned)

        # 1. Transform Target Feature (Price) via log1p transformation
        print("\n📈 Step 4: Stabilizing target skewness with natural log scaling...")
        df_standardized["log_price"] = np.log1p(df_standardized["price"])
        y = df_standardized["log_price"]
        X = df_standardized.drop(columns=["price", "log_price", "date_added_clean"], errors="ignore")

        # 2. Holdout Partitioning (Train/Test Split)
        print("\n✂️ Step 5: Segmenting data into 80/20 Train and Test matrices...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Fit Target Encoding strictly on training bounds to block data leakage
        if "location" in X_train.columns:
            print("\n🗺️ Step 6: Target-Encoding high-cardinality location footprints...")
            train_temp = X_train.copy()
            train_temp['target_log_price'] = y_train
            
            # Create mapping files
            self.location_encoder_map = train_temp.groupby('location')['target_log_price'].mean().to_dict()
            self.global_log_price_mean = y_train.mean()

            # Map encoder transformations across splits safely
            X_train['location_encoded'] = X_train['location'].map(self.location_encoder_map).fillna(self.global_log_price_mean)
            X_test['location_encoded'] = X_test['location'].map(self.location_encoder_map).fillna(self.global_log_price_mean)
            
            X_train = X_train.drop(columns=['location'])
            X_test = X_test.drop(columns=['location'])

        # 4. Apply Categorical One-Hot Encoding Expansion
        print("\n🔠 Step 7: Performing One-Hot Encoding expansions...")
        cat_cols = ["property_type", "city", "province_name"]
        cat_cols = [c for c in cat_cols if c in X_train.columns]

        X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        
        # Align structural columns across both spaces
        X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)
        self.training_feature_columns = X_train_enc.columns.tolist()

        # 5. Apply Feature Scaling Standardizations
        print("\n📐 Step 8: Scaling mathematical values using StandardScaler...")
        scale_targets = [c for c in (self.continuous_cols + self.spatial_cols) if c in X_train_enc.columns]
        
        X_train_scaled = X_train_enc.copy()
        X_test_scaled = X_test_enc.copy()

        X_train_scaled[scale_targets] = self.scaler.fit_transform(X_train_enc[scale_targets])
        X_test_scaled[scale_targets] = self.scaler.transform(X_test_enc[scale_targets])

        # 6. Save Preprocessed Data Sheets
        print("\n💾 Step 9: Exporting processed matrices to disk space...")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        train_out = X_train_scaled.copy()
        train_out["target_log_price"] = y_train
        test_out = X_test_scaled.copy()
        test_out["target_log_price"] = y_test

        train_out.to_csv(os.path.join(self.processed_dir, "zameen_train_preprocessed.csv"), index=False)
        test_out.to_csv(os.path.join(self.processed_dir, "zameen_test_preprocessed.csv"), index=False)

        # 7. Persist Pipeline Artifacts for Production Use
        self.save_pipeline_artifacts()
        
        print("✅ End-to-End Data Pipeline Executed and Persisted Successfully.")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_pipeline_artifacts(self):
        """Serializes and saves pipeline parameters to serve live incoming user queries later."""
        print("\n📦 Step 10: Serializing pipeline objects for production scaling artifacts...")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        artifacts = {
            "scaler": self.scaler,
            "location_map": self.location_encoder_map,
            "global_mean": self.global_log_price_mean,
            "feature_columns": self.training_feature_columns
        }
        
        artifact_path = os.path.join(self.artifacts_dir, "data_pipeline_assets.joblib")
        joblib.dump(artifacts, artifact_path)
        print(f"   📥 Successfully archived pipeline models inside: {artifact_path}")


# --- Execution Controller Window ---
if __name__ == "__main__":
    # Define production root project paths
    RAW_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Raw\zameen-updated.csv"
    PROCESSED_DIR_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed"
    ARTIFACTS_DIR_PATH = r"D:\Data Science Projects\Property-Price-Detection\models"

    # Instantiate and fire pipeline process execution
    pipeline = ZameenDataPipeline(
        raw_data_path=RAW_PATH, 
        processed_dir=PROCESSED_DIR_PATH, 
        artifacts_dir=ARTIFACTS_DIR_PATH
    )
    X_train, X_test, y_train, y_test = pipeline.execute_pipeline()
