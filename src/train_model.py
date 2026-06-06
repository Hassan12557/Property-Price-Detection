import os
import numpy as np
import pandas as pd

# Import Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Import Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_real_estate_models(train_path, test_path):
    """
    Trains 6 different regression models, evaluates them using R2, MAE, and RMSE,
    and returns a side-by-side performance breakdown.
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Preprocessed data files missing. Please run data_preprocessing.py first.")

    print("📥 Loading preprocessed training and testing datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features (X) and target (y)
    # Note: We drop 'date_added_clean' because raw date strings cannot be parsed by model math
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

    print("\n🚀 Training models and calculating metrics...")
    for name, model in models.items():
        print(f"   🔄 Training {name}...")
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set (Outputs are in Log scale)
        log_preds = model.predict(X_test)
        
        # Inverse transform log values back to raw Pakistani Rupees (PKR)
        actual_pkr = np.expm1(y_test)
        pred_pkr = np.expm1(log_preds)
        
        # Calculate evaluation metrics based on real PKR values
        r2 = r2_score(y_test, log_preds) # R2 remains identical on log or exponential scale
        mae_pkr = mean_absolute_error(actual_pkr, pred_pkr)
        rmse_pkr = np.sqrt(mean_squared_error(actual_pkr, pred_pkr))
        
        results.append({
            "Model Name": name,
            "R2 Score (Explainer)": f"{r2:.4f}",
            "MAE (Avg Error in PKR)": f"{mae_pkr:,.2f}",
            "RMSE (Outlier Error PKR)": f"{rmse_pkr:,.2f}"
        })

    # Convert results into a structured DataFrame for clean visual comparison
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(by="R2 Score (Explainer)", ascending=False).reset_index(drop=True)
    
    return summary_df


if __name__ == "__main__":
    # Define exact paths to your preprocessed data folders
    TRAIN_PREPROCESSED_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed\zameen_train_preprocessed.csv"
    TEST_PREPROCESSED_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed\zameen_test_preprocessed.csv"

    performance_table = evaluate_real_estate_models(TRAIN_PREPROCESSED_PATH, TEST_PREPROCESSED_PATH)
    
    print("\n" + "="*75)
    print("🏆 FINAL MODEL PERFORMANCE LEADERBOARD")
    print("="*75)
    print(performance_table.to_string())
    print("="*75)
