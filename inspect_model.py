import joblib
import pandas as pd

# Load our trained model
model = joblib.load("models/spam_detector_unique.joblib")

# Get the TF-IDF vectorizer
vectorizer = model.named_steps["tfidf"]

# Get the trained Linear SVM classifier
classifier = model.named_steps["classifier"]

# Get all words and phrases learned by the model
features = vectorizer.get_feature_names_out()

# Get the importance scores
weights = classifier.coef_[0]

# Combine words and their scores
feature_data = pd.DataFrame({
    "Word": features,
    "Score": weights
})

# Words with the highest positive scores
spam_words = feature_data.sort_values(
    by="Score",
    ascending=False
).head(20)

# Words with the lowest negative scores
ham_words = feature_data.sort_values(
    by="Score",
    ascending=True
).head(20)

print("\n--- WORDS MOST ASSOCIATED WITH SPAM ---")
print(spam_words.to_string(index=False))

print("\n--- WORDS MOST ASSOCIATED WITH HAM ---")
print(ham_words.to_string(index=False))