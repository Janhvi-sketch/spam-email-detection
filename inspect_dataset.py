
import pandas as pd

# Load the dataset
file_path = "data/raw/enron_spam/enron_spam_data.csv"

df = pd.read_csv(file_path, encoding="latin-1")

# 1. Dataset size
print("\n--- DATASET SIZE ---")
print("Rows and columns:", df.shape)

# 2. Column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 3. Data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 4. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 5. Duplicate rows
print("\n--- DUPLICATES ---")
print("Duplicate rows:", df.duplicated().sum())

# 6. Spam and legitimate emails
print("\n--- EMAIL DISTRIBUTION ---")
print(df["Spam/Ham"].value_counts())

# 7. Email length
print("\n--- EMAIL LENGTH ---")
print("Average subject length:",
      df["Subject"].fillna("").astype(str).str.len().mean())

print("Average message length:",
      df["Message"].fillna("").astype(str).str.len().mean())


# Check incomplete emails

subject_missing = (
    df["Subject"].isna() |
    df["Subject"].fillna("").astype(str).str.strip().eq("")
)

message_missing = (
    df["Message"].isna() |
    df["Message"].fillna("").astype(str).str.strip().eq("")
)

print("\n--- INCOMPLETE EMAIL ANALYSIS ---")

print("Missing subject:", subject_missing.sum())
print("Missing message:", message_missing.sum())

print("Missing both:", (subject_missing & message_missing).sum())

print(
    "Only subject missing:",
    (subject_missing & ~message_missing).sum()
)

print(
    "Only message missing:",
    (~subject_missing & message_missing).sum()
)