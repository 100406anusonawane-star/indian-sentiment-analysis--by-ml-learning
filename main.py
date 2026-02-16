# Simple Sentiment Analysis Project

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("data/reviews.csv")

# Clean sentiment labels
data['sentiment'] = data['sentiment'].str.strip().str.lower()
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Remove invalid rows if any
data = data.dropna(subset=['sentiment'])

print("Sentiment distribution:")
print(data['sentiment'].value_counts())

# Text cleaning
stop_words = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['cleaned_review'] = data['review'].apply(clean_text)

# Split data
X = data['cleaned_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Results
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom test
sample = "Product quality bahut achi hai"
sample_clean = clean_text(sample)
sample_vec = vectorizer.transform([sample_clean])
prediction = model.predict(sample_vec)

print("\nCustom Review Prediction:")
print(sample, "→", "Positive" if prediction[0] == 1 else "Negative")