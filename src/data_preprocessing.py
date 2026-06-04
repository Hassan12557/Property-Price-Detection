import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import skew


def check_price_distribution(file_path, price_column_name):
    """Loads the Zameen dataset and checks the balance/skewness of the price column."""
    # 1. Verify file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find the file at: {file_path}. Please check the path!"
        )

    print("🔄 Loading dataset...")
    # Read the data (handles both CSV and Excel based on extension)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "Unsupported file format. Please ensure your file path ends with .csv, .xlsx, or .xls"
        )

    print(f"✅ Data loaded successfully. Shape: {df.shape}")

    # 2. Check if the specified price column exists
    if price_column_name not in df.columns:
        raise KeyError(
            f"The column '{price_column_name}' was not found in the dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop missing values in the price column just for calculation and visualization
    price_data = df[price_column_name].dropna()

    # 3. Calculate Mathematical Skewness
    price_skewness = skew(price_data)

    print("\n" + "=" * 40)
    print(f"📊 DISTRIBUTION CHECK FOR: '{price_column_name}'")
    print("=" * 40)
    print(f"Calculated Skewness Score: {price_skewness:.2f}")

    if price_skewness > 1:
        print(
            "Status: UNBALANCED (Highly Right-Skewed). Data contains a long tail of high-end properties."
        )
    elif price_skewness < -1:
        print(
            "Status: UNBALANCED (Highly Left-Skewed). Data contains a long tail of low-end properties."
        )
    else:
        print(
            "Status: BALANCED (Symmetric). The data resembles a normal distribution curve."
        )
    print("=" * 40 + "\n")

    # 4. Visualize the Distribution
    print("🎨 Generating distribution plot...")
    plt.figure(figsize=(9, 6))

    sns.histplot(price_data, kde=True, color="crimson", bins=50)

    plt.title(
        f"Raw Price Distribution Check\n(Skewness Score: {price_skewness:.2f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Price (PKR)", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


# --- Execution Block ---
if __name__ == "__main__":
    # 1. Added 'r' for a raw string to handle Windows backslashes safely
    # 2. ⚠️ MAKE SURE TO ADD YOUR FILE EXTENSION AT THE END (e.g., .csv or .xlsx)
    CHOSEN_FILE_PATH = r"D:\Data Science Projects\Property-Price-Detection\Data\Raw\zameen-updated.csv"

    # Match this exactly to your dataset's price header
    TARGET_COLUMN = "price"

    # Run the validation
    check_price_distribution(CHOSEN_FILE_PATH, TARGET_COLUMN)