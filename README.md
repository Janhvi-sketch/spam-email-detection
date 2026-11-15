
# 📧 Spam Email Detection System

A Machine Learning project that classifies emails as Spam or Ham (Legitimate) using Natural Language Processing and TF-IDF.

## Features

- Email spam classification
- TF-IDF text processing
- Linear Support Vector Machine (Linear SVM)
- Probability estimates using calibrated classification
- Interactive Streamlit web application
- Model evaluation with multiple performance metrics

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Linear SVM
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Model Performance

Evaluated on 6,028 unseen test emails.

| Metric | Score |
|---|---:|
| Accuracy | 98.69% |
| Brier Score | 0.0201 |
| Average Precision | 99.86% |

## Project Structure

```text
spamemail/
├── data/
│   ├── raw/
│   └── processed/
├── evaluation/
├── models/
├── app.py
├── clean_data.py
├── compare_models.py
├── evaluate_model.py
├── predict_email.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Run Locally

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the application:

```bash
streamlit run app.py
```

## Dataset

This project uses the Enron Spam dataset.

## Limitations

The model may learn patterns specific to the Enron dataset. Performance on emails from different sources may vary. Probability estimates are not guaranteed to be perfectly calibrated.

## Author

Janhvi Morajkar
