# 🎭 Emotion Detector

A Streamlit web app that classifies text into one of six emotions — **sadness, anger, love, surprise, fear, joy** — using a TF-IDF + Logistic Regression model. Type a sentence, and the app predicts the emotion and shows confidence scores in an interactive chart.

## Features

- Predicts one of six emotions from free-form text
- Interactive bar chart and table of confidence scores for every emotion
- Optional view of the cleaned text sent to the model
- Lightweight: stopwords are hardcoded, so no NLTK downloads at runtime

## Project Structure

```
.
├── app.py                                        # Streamlit app
├── NLP_Project_Sentiment_Analysis.ipynb          # Training & evaluation notebook
├── logistic_regression_tfidf_model (1).joblib    # Trained Logistic Regression model
├── tfidf_vectorizer.joblib                       # Fitted TF-IDF vectorizer
├── train_file.txt                                # Training dataset (text;emotion)
├── requirements.txt                              # Python dependencies
└── README.md
```

> **Note:** `app.py` loads the model from `logistic_regression_tfidf_model (1).joblib`. Make sure the filename on disk matches the path in `app.py`.

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501), enter some text, and click **Analyze Emotion**.

**Example:** `I can't believe how wonderful today turned out to be!` → 😄 Joy

## How It Works

Input text goes through the same preprocessing used during training:

1. Lowercase
2. Remove punctuation
3. Remove numbers
4. Remove non-ASCII characters (e.g. emojis)
5. Remove stopwords (standard NLTK English list)
6. Convert to features with `TfidfVectorizer`
7. Predict with `LogisticRegression`

## Dataset

The model was trained on a labeled emotion dataset of 16,000 short texts, stored as `text;emotion` per line. Labels map as follows:

| Label | Emotion  |
|-------|----------|
| 0     | Sadness  |
| 1     | Anger    |
| 2     | Love     |
| 3     | Surprise |
| 4     | Fear     |
| 5     | Joy      |

## Model Performance

Evaluated on a held-out test set of 3,200 samples:

| Metric            | Score |
|-------------------|-------|
| Accuracy          | 86.3% |
| Macro avg F1      | 0.80  |
| Weighted avg F1   | 0.86  |

The model performs best on the larger classes (sadness, joy, anger) and is weaker on the rarer ones — **surprise** (F1 0.61) and **love** (F1 0.73) — which have fewer training examples.

## Tech Stack

- Python
- scikit-learn (TF-IDF, Logistic Regression)
- Streamlit
- pandas
- Altair

## Limitations

- Predictions reflect patterns in the training data and may not always be accurate.
- Short or ambiguous text, sarcasm, and mixed emotions can be misclassified.
- Input that becomes empty after cleaning (only punctuation or stopwords) can't be analyzed.
- The model files were saved with scikit-learn 1.6.1; use the pinned version in `requirements.txt`.
