
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/emails_cleaned.csv")

print("Dataset loaded:", df.shape)


# 2. Combine subject and message
df["Email Text"] = (
    df["Subject"].fillna("") + " " +
    df["Message"].fillna("")
)


# 3. Separate input and output
X = df["Email Text"]
y = df["Spam/Ham"]


# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 5. Display split information
print("\n--- DATA SPLIT ---")

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())
