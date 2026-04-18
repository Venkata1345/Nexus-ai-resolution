import pandas as pd
from datasets import load_dataset
import os

print("Downloading Bitext dataset from Hugging Face...")

# Load the dataset directly via the datasets library
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Convert the 'train' split to a Pandas DataFrame
df = dataset['train'].to_pandas()

# Ensure the path uses the correct slashes for your OS
output_path = os.path.join("data", "raw", "bitext_support_data.csv")

# Save it to our raw data folder
df.to_csv(output_path, index=False)

print(f"Success! Saved {len(df)} records to {output_path}")