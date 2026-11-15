
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score
)

from sklearn.calibration import calibration_curve


# 1. Create evaluation folder
os.makedirs("evaluation", exist_ok=True)


# 2. Load the test dataset
test_data = pd.read_csv(
    "data/processed/test_unique.csv"
)

X_test = test_data["Email Text"]
y_test = test_data["Spam/Ham"]


# 3. Load the trained model
model = joblib.load(
    "models/spam_detector_unique.joblib"
)


# 4. Make predictions
y_pred = model.predict(X_test)

# Get probability of spam
classes = list(model.classes_)
spam_index = classes.index("spam")

y_prob = model.predict_proba(X_test)[:, spam_index]


# 5. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

brier = brier_score_loss(
    y_test,
    y_prob,
    pos_label="spam"
)

avg_precision = average_precision_score(
    y_test == "spam",
    y_prob
)


# 6. Print results
print("\n--- MODEL EVALUATION ---")

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

print("\n--- CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"]
    )
)


# 7. Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Spam"]
)

display.plot(cmap="Blues", values_format="d")

plt.title("Spam Email Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "evaluation/confusion_matrix.png",
    dpi=300
)

plt.close()


# 8. Calibration Curve
actual_spam = (y_test == "spam").astype(int)

fraction_positive, mean_predicted = calibration_curve(
    actual_spam,
    y_prob,
    n_bins=10,
    strategy="uniform"
)

plt.figure(figsize=(7, 5))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Spam Detector"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Actual Spam Frequency")
plt.title("Probability Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "evaluation/calibration_curve.png",
    dpi=300
)

plt.close()


# 9. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(
    actual_spam,
    y_prob
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {avg_precision:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Spam Detection - Precision Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "evaluation/precision_recall_curve.png",
    dpi=300
)

plt.close()


# 10. Save evaluation metrics
with open("evaluation/metrics.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Brier Score: {brier:.4f}\n")
    file.write(f"Average Precision: {avg_precision:.4f}\n")
    file.write("\nClassification Report:\n")
    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"]
        )
    )

print("\nEvaluation completed!")
print("Charts and metrics saved in the evaluation folder.")
