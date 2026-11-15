
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/emails_cleaned.csv")

# 2. Combine subject and message
df["Email Text"] = (
    df["Subject"].fillna("") + " " +
    df["Message"].fillna("")
)

# 3. Normalize text for duplicate checking
normalized_text = (
    df["Email Text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 4. Count duplicate email content
duplicate_rows = normalized_text.duplicated(keep=False).sum()

duplicate_groups = normalized_text[
    normalized_text.duplicated(keep=False)
].nunique()

print("--- DUPLICATE CONTENT CHECK ---")
print("Rows with repeated email content:", duplicate_rows)
print("Number of repeated content groups:", duplicate_groups)

# 5. Recreate the same train/test split
X = normalized_text
y = df["Spam/Ham"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Check whether identical emails occur in both sets
train_emails = set(X_train)
test_emails = set(X_test)

overlap = train_emails.intersection(test_emails)
overlap.discard("")

print("\n--- TRAIN/TEST LEAKAGE CHECK ---")
print("Identical emails in both sets:", len(overlap))

if len(overlap) == 0:
    print("No exact email-content overlap detected.")
else:
    print("Potential data leakage detected!")

