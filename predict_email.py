
import joblib

# Load the calibrated model
model = joblib.load("models/spam_detector_unique.joblib")


def predict_email(email_text):

    # Check empty input
    if not email_text or not email_text.strip():
        return None

    # Predict email category
    prediction = model.predict([email_text])[0]

    # Get probability estimates
    probabilities = model.predict_proba([email_text])[0]

    # Find probability belonging to predicted category
    classes = model.classes_
    confidence = probabilities[list(classes).index(prediction)]

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    }


# Run from terminal
if __name__ == "__main__":

    email = input("Paste your email here: ")

    result = predict_email(email)

    if result is None:
        print("Please enter an email!")

    else:
        print("\n--- EMAIL PREDICTION ---")
        print("Prediction:", result["prediction"].upper())
        print("Estimated probability:", result["confidence"], "%")
