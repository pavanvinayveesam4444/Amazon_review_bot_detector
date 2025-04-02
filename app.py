import streamlit as st
import joblib
import re, string

# Load model & vectorizer
model = joblib.load("bot_detector_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Clean review
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

# Streamlit UI
st.title("🕵️ Amazon Review Bot Detector")
st.write("Paste your Amazon review below to detect whether it was written by a human or a bot.")

review = st.text_area("✍️ Enter review text:", height=200)

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        pred_class = int(prediction)
        confidence = round(proba[pred_class] * 100, 2)

        label = "🤖 Bot" if pred_class == 1 else "🧑 Human"
        st.markdown(f"### Result: {label}")
        st.markdown(f"**Confidence:** {confidence}%")