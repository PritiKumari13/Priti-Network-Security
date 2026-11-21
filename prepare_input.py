
import glob
import os
import pandas as pd

# Find latest ingested test dataset
files = glob.glob(
    "Artifacts/*/data_ingestion/ingested/test.csv"
)

if not files:
    raise FileNotFoundError("Ingested test.csv not found.")

latest_file = max(files, key=os.path.getmtime)

# Load dataset
df = pd.read_csv(latest_file)

print("Original dataset shape:", df.shape)

# Remove target column
if "Result" in df.columns:
    df = df.drop(columns=["Result"])

# Select first 5 records
sample_df = df.head(5).copy()

# Create output folder
os.makedirs("prediction_output", exist_ok=True)

# Save prediction input
output_file = "prediction_output/sample_input.csv"

sample_df.to_csv(output_file, index=False)

print("\nPrediction CSV created successfully!")
print("File:", os.path.abspath(output_file))
print("Input shape:", sample_df.shape)
print("Number of features:", len(sample_df.columns))
print("\nPreview:")
print(sample_df.head())
