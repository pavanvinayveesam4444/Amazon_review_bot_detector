import pandas as pd
import re, string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load dataset
df = pd.read_json("Electronics.json", lines=True)
df = df[['reviewText', 'class']].dropna()

# Sampling logic
df_0 = df[df['class'] == 0].sample(n=100000, random_state=42)
df_1 = df[df['class'] == 1].sample(n=20000, random_state=42)
df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

df['cleaned'] = df['reviewText'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['class'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "bot_detector_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved!")