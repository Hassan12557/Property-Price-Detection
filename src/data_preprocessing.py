import os
import pandas as pd

file_path= 'D:\Data Science Projects\Property-Price-Detection\Data\Raw\zameen-updated.csv'
def import_data(file_path: str) -> pd.DataFrame:
    """Imports data from a specified file path based on its extension.

    Supported formats: .csv, .xlsx, .json
    """
    # Check if the file actually exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' was not found.")

    # Extract the file extension
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    print(f"Attempting to import: {os.path.basename(file_path)}")

    # Read data based on file type
    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_extension == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. Please use CSV, Excel, or JSON."
        )

    print(f"Successfully imported! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    return df


if __name__ == "__main__":
    # --- PyCharm Configuration ---
    # Replace this path with the actual path to your dataset.
    # Tip: In PyCharm, you can right-click your data file and select "Copy Path/Reference..."
    DATA_PATH = "data/your_dataset.csv"

    try:
        # Load the dataset
        data = import_data(DATA_PATH)

        # Quick preview of the data in the PyCharm console
        print("--- First 5 Rows ---")
        print(data.head())

    except Exception as e:
        print(f"Error during import: {e}")
