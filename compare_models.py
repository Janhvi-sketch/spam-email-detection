
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


# 1. Load training data only
df = pd.read_csv("data/processed/train_unique.csv")

X = df["Email Text"]
y = df["Spam/Ham"]

print("Training emails:", len(X))


# 2. Define evaluation metrics
scoring = {
    "precision": make_scorer(
        precision_score,
        pos_label="spam"
    ),
    "recall": make_scorer(
        recall_score,
        pos_label="spam"
    ),
    "f1": make_scorer(
        f1_score,
        pos_label="spam"
    )
}


# 3. Prepare cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 4. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC()
}


# 5. Compare models
results = []

for name, classifier in models.items():

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        ),
        ("classifier", classifier)
    ])

    print("\nEvaluating:", name)

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1 Score": scores["test_f1"].mean()
    })


# 6. Display comparison
results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(
    results_df.sort_values(
        by="F1 Score",
        ascending=False
    ).to_string(index=False)
)
