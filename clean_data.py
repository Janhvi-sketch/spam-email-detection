
import pandas as pd

# 1. Load the original dataset
file_path = "data/raw/enron_spam/enron_spam_data.csv"

df = pd.read_csv(file_path, encoding="latin-1")

print("Original dataset shape:", df.shape)


# 2. Check missing subject and message values
subject_missing = (
    df["Subject"].isna() |
    df["Subject"].fillna("").astype(str).str.strip().eq("")
)

message_missing = (
    df["Message"].isna() |
    df["Message"].fillna("").astype(str).str.strip().eq("")
)


# 3. Remove emails with both subject and message missing
df_clean = df.loc[~(subject_missing & message_missing)].copy()

print("Rows after removing empty emails:", df_clean.shape)


# 4. Fill missing subjects and messages with empty strings
df_clean["Subject"] = df_clean["Subject"].fillna("")
df_clean["Message"] = df_clean["Message"].fillna("")


# 5. Remove unnecessary spaces around text
df_clean["Subject"] = df_clean["Subject"].astype(str).str.strip()
df_clean["Message"] = df_clean["Message"].astype(str).str.strip()


# 6. Check the cleaned dataset
print("\n--- CLEANED DATASET ---")
print("Final shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean[["Subject", "Message", "Spam/Ham"]].isnull().sum())

print("\nEmail distribution:")
print(df_clean["Spam/Ham"].value_counts())


# 7. Save cleaned dataset
output_path = "data/processed/emails_cleaned.csv"

import os
os.makedirs("data/processed", exist_ok=True)

df_clean.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)
