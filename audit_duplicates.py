
import pandas as pd

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/emails_cleaned.csv")

# 2. Combine subject and message
df["Email Text"] = (
    df["Subject"].fillna("") + " " +
    df["Message"].fillna("")
)

# 3. Normalize email text
df["Normalized Text"] = (
    df["Email Text"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 4. Remove empty email content from this audit
df = df[df["Normalized Text"] != ""].copy()

# 5. Group identical emails and count their labels
label_counts = (
    df.groupby("Normalized Text")["Spam/Ham"]
    .value_counts()
    .unstack(fill_value=0)
)

# 6. Find repeated email groups
group_sizes = df.groupby("Normalized Text").size()

repeated_groups = group_sizes[group_sizes > 1]

# 7. Find groups with conflicting labels
conflicting_groups = label_counts[
    (label_counts > 0).sum(axis=1) > 1
]

# 8. Display results
print("--- DUPLICATE AUDIT ---")

print("Unique non-empty email contents:", len(group_sizes))

print("Repeated email groups:", len(repeated_groups))

print(
    "Repeated groups with conflicting labels:",
    len(conflicting_groups)
)

print(
    "Emails belonging to conflicting groups:",
    int(conflicting_groups.sum(axis=1).sum())
)

print("\n--- LABEL BREAKDOWN ---")
print(label_counts.sum())

print("\n--- CONFLICTING LABEL EXAMPLES ---")

if len(conflicting_groups) > 0:
    print(conflicting_groups.head(5))
else:
    print("No conflicting labels found.")
