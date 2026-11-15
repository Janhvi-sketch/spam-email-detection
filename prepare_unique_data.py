
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/emails_cleaned.csv")

print("Original dataset:", df.shape)


# 2. Combine subject and message
df["Email Text"] = (
    df["Subject"].fillna("") + " " +
    df["Message"].fillna("")
)


# 3. Normalize text for duplicate detection
df["Normalized Text"] = (
    df["Email Text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# 4. Keep only one copy of each email
unique_df = df.drop_duplicates(
    subset=["Normalized Text"],
    keep="first"
).copy()

print("\n--- DUPLICATE REMOVAL ---")
print("Original emails:", len(df))
print("Unique emails:", len(unique_df))
print("Duplicates removed:", len(df) - len(unique_df))


# 5. Separate features and labels
X = unique_df["Email Text"]
y = unique_df["Spam/Ham"]


# 6. Split unique emails
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 7. Save training and testing datasets
train_df = pd.DataFrame({
    "Email Text": X_train,
    "Spam/Ham": y_train
})

test_df = pd.DataFrame({
    "Email Text": X_test,
    "Spam/Ham": y_test
})

train_df.to_csv("data/processed/train_unique.csv", index=False)
test_df.to_csv("data/processed/test_unique.csv", index=False)


# 8. Verify no overlap
train_normalized = set(
    X_train.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)

test_normalized = set(
    X_test.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
)

overlap = train_normalized.intersection(test_normalized)

print("\n--- NEW DATA SPLIT ---")
print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())

print("\nIdentical emails in both sets:", len(overlap))

print("\nDatasets saved successfully!")
