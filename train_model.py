
import pandas as pd
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# STEP 1: Load the new datasets
train_df = pd.read_csv("data/processed/train_unique.csv")
test_df = pd.read_csv("data/processed/test_unique.csv")

X_train = train_df["Email Text"]
y_train = train_df["Spam/Ham"]

X_test = test_df["Email Text"]
y_test = test_df["Spam/Ham"]

print("Training emails:", len(X_train))
print("Testing emails:", len(X_test))


# STEP 2: Create the machine learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
    "classifier",
    CalibratedClassifierCV(
        estimator=LinearSVC(),
        method="sigmoid",
        cv=5
    )
)
])


# STEP 3: Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# STEP 4: Predict test emails
y_pred = model.predict(X_test)


# STEP 5: Evaluate the model
print("\n--- MODEL EVALUATION ---")

print(
    "Accuracy:",
    round(accuracy_score(y_test, y_pred) * 100, 2),
    "%"
)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["ham", "spam"]
    )
)


# STEP 6: Save the model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/spam_detector_unique.joblib"
)

print("\nModel saved successfully!")
