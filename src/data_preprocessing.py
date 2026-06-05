import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import skew


def preprocess_data(file_path, output_path):
    """
    Cleans the dataset by dropping unnecessary columns, filling missing cells,
    and standardizing the date format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the raw file at: {file_path}")

    print("🛠️ Starting Data Preprocessing Pipeline...")
    
    # 1. Load Data
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    
    print(f"📋 Initial Shape: {df.shape}")

    # 2. Drop unnecessary columns (e.g., page_url)
    col_to_drop = "page_url"
    if col_to_drop in df.columns:
        df = df.drop(columns=[col_to_drop])
        print(f"🗑️ Successfully dropped column: '{col_to_drop}'")
    else:
        print(f"ℹ️ Column '{col_to_drop}' not found or already dropped.")

    # 3. Handle Empty Cells (Missing Values)
    print("🩹 Handling missing values...")
    missing_counts = df.isnull().sum()
    columns_with_nas = missing_counts[missing_counts > 0]
    
    if len(columns_with_nas) > 0:
        print("🔍 Found missing values in the following columns:")
        for col, count in columns_with_nas.items():
            print(f"   - {col}: {count} empty rows")
        
        # Apply specific filling strategy based on data type
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric missing rows with median to safeguard against outliers
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    # Fill categorical missing rows with a standard placeholder
                    df[col] = df[col].fillna("Unknown")
        print("✅ All empty cells filled successfully.")
    else:
        print("✨ Clean data record check: No empty cells found.")

    # 4. Fix and Standardize 'date_added' Formats
    # Handles variations like 'dateadded', 'Date Added', or 'date_added'
    date_col = None
    for alternative in ["date_added", "dateadded", "Date Added"]:
        if alternative in df.columns:
            date_col = alternative
            break

    if date_col:
        print(f"📅 Standardizing date formats in column: '{date_col}'")
        # Rename column to a clean snake_case format for consistency
        if date_col != "date_added":
            df = df.rename(columns={date_col: "date_added"})
            date_col = "date_added"
            
        # Convert to datetime object (errors='coerce' turns unparseable formats to NaT)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Fill missing dates with the most frequent date (mode)
        if df[date_col].isnull().sum() > 0:
            df[date_col] = df[date_col].fillna(df[date_col].mode()[0])
            
        # Standardize format display to YYYY-MM-DD
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
        print(f"✅ Dates standardized to YYYY-MM-DD style.")
    else:
        print("⚠️ Warning: No date tracking reference matching 'date_added' found.")

    # 5. Save the processed data file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned dataset exported here: {output_path}")
    print(f"📋 Final Cleaned Shape: {df.shape}\n" + "-"*50)
    
    return output_path


def check_price_distribution(file_path, price_column_name):
    """Loads the Zameen dataset and checks the balance/skewness of the price column."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)

    if price_column_name not in df.columns:
        raise KeyError(f"The column '{price_column_name}' was not found.")

    price_data = df[price_column_name].dropna()
    price_skewness = skew(price_data)

    print("\n" + "=" * 40)
    print(f"📊 DISTRIBUTION CHECK FOR: '{price_column_name}'")
    print("=" * 40)
    print(f"Calculated Skewness Score: {price_skewness:.2f}")

    if price_skewness > 1:
        print("Status: UNBALANCED (Highly Right-Skewed).")
    elif price_skewness < -1:
        print("Status: UNBALANCED (Highly Left-Skewed).")
    else:
        print("Status: BALANCED (Symmetric).")
    print("=" * 40 + "\n")

    print("🎨 Generating distribution plot...")
    plt.figure(figsize=(9, 6))
    sns.histplot(price_data, kde=True, color="crimson", bins=50)
    plt.title(f"Raw Price Distribution Check\n(Skewness Score: {price_skewness:.2f})", fontweight="bold")
    plt.xlabel("Price (PKR)")
    plt.ylabel("Frequency (Count)")
    plt.tight_layout()
    plt.show()


def check_categorical_balance(file_path, categorical_columns):
    """Loads the dataset and prints the percentage representation/balance of categorical columns."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)

    print("\n" + "=" * 50)
    print("📊 CATEGORICAL FEATURE REPRESENTATION & BALANCE CHECK")
    print("=" * 50)

    for col in categorical_columns:
        if col in df.columns:
            print(f"\n🔹 Feature: '{col}'")
            print("-" * 40)
            counts = df[col].value_counts(dropna=False)
            percentages = df[col].value_counts(normalize=True, dropna=False) * 100
            summary_df = pd.DataFrame({"Count": counts, "Percentage (%)": percentages.round(2)})
            print(summary_df.head(10))

            top_class_pct = percentages.iloc[0]
            if top_class_pct > 85:
                print(f"⚠️ WARNING: '{col}' is heavily imbalanced! '{percentages.index[0]}' occupies {top_class_pct:.1f}%.")
        else:
            print(f"\n❌ Column '{col}' not detected.")
    print("\n" + "=" * 50 + "\n")


# --- Execution Block ---
if __name__ == "__main__":
    # Path configuration setup
    RAW_FILE_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Raw\zameen-updated.csv"
    PROCESSED_FILE_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Processed\zameen_cleaned.csv"

    # Step 1: Run the Preprocessing Pipeline first
    clean_data_path = preprocess_data(RAW_FILE_PATH, PROCESSED_FILE_PATH)

    # Step 2: Run validation checks on the newly cleaned dataset
    TARGET_COLUMN = "price"
    check_price_distribution(clean_data_path, TARGET_COLUMN)

    CATEGORICAL_COLS = [
        "property_type",
        "city",
        "province_name",
        "location",
        "purpose",
        "Area Type",
        "Area Category",
    ]
    check_categorical_balance(clean_data_path, CATEGORICAL_COLS)