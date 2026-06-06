import os
import joblib
import numpy as np
import pandas as pd

# Import Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Import Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_evaluate_and_save_best_model(train_path, test_path, models_dir):
    """
    Trains 6 regression models, evaluates performance, prints a leaderboard,
    and automatically saves the highest-scoring model to a .pkl file.
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Preprocessed data files missing. Please run data_pipeline.py first.")

    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)

    print("📥 Loading preprocessed training and testing datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features (X) and target (y)
    # Dropping admin/target columns before training
    drop_cols = ['target_log_price', 'date_added_clean']
    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['target_log_price']
    
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    y_test = test_df['target_log_price']

    # Initialize the 6 algorithms
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=15, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(random_state=42, n_estimators=50),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
        "XGBoost": XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1, n_jobs=-1)
    }

    results = []
    trained_model_objects = {}

    print("\n🚀 Training models and calculating metrics...")
    for name, model in models.items():
        print(f"   🔄 Training {name}...")
        # Train the model
        model.fit(X_train, y_train)
        
        # Keep track of the trained object in memory
        trained_model_objects[name] = model
        
        # Predict on the test set (Outputs are in Log scale)
        log_preds = model.predict(X_test)
        
        # Inverse transform log values back to raw Pakistani Rupees (PKR)
        actual_pkr = np.expm1(y_test)
        pred_pkr = np.expm1(log_preds)
        
        # Calculate evaluation metrics based on real PKR values
        r2 = r2_score(y_test, log_preds)
        mae_pkr = mean_absolute_error(actual_pkr, pred_pkr)
        rmse_pkr = np.sqrt(mean_squared_error(actual_pkr, pred_pkr))
        
        results.append({
            "Model Name": name,
            "R2_Score_Raw": r2,  # Keep as a raw float for sorting logic
            "R2 Score (Explainer)": f"{r2:.4f}",
            "MAE (Avg Error in PKR)": f"{mae_pkr:,.2f}",
            "RMSE (Outlier Error PKR)": f"{rmse_pkr:,.2f}"
        })

    # Convert results into a structured DataFrame for clean visual comparison
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(by="R2_Score_Raw", ascending=False).reset_index(drop=True)
    
    # Identify the champion model dynamically
    best_model_name = summary_df.loc[0, "Model Name"]
    best_model_r2 = summary_df.loc[0, "R2 Score (Explainer)"]
    best_model_obj = trained_model_objects[best_model_name]
    
    # Drop the raw numeric float column before printing to keep the leaderboard clean
    summary_df = summary_df.drop(columns=["R2_Score_Raw"])
    
    print("\n" + "="*75)
    print("🏆 FINAL MODEL PERFORMANCE LEADERBOARD")
    print("="*75)
    print(summary_df.to_string())
    print("="*75)

    # --- AUTOMATED MODEL SERIALIZATION ---
    print(f"\n💾 Step 10: Saving champion model architecture...")
    model_save_path = os.path.join(models_dir, "model.pkl")
    
    # Save the trained sklearn/xgboost object to disk
    joblib.dump(best_model_obj, model_save_path)
    
    print(f"   🥇 Success! '{best_model_name}' (R2: {best_model_r2}) has been saved as the production checkpoint.")
    print(f"   📥 Destination File: {model_save_path}")


if __name__ == "__main__":
    # Define production root project paths matching your local environment layout
    TRAIN_PREPROCESSED_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed\zameen_train_preprocessed.csv"
    TEST_PREPROCESSED_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed\zameen_test_preprocessed.csv"
    MODELS_DIR_PATH = r"D:\Data Science Projects\Property-Price-Detection\models"

    train_evaluate_and_save_best_model(
        TRAIN_PREPROCESSED_PATH, 
        TEST_PREPROCESSED_PATH, 
        MODELS_DIR_PATH
    )
