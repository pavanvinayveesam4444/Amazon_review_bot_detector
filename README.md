# Review Bot Detector

## Project Overview

This project leverages Machine Learning and  Text Mining Techniques to build a spam classifier that identifies whether an Amazon product review was written by a human or a bot.
The classifier is trained using real Amazon Electronics reviews and deployed using  Streamlit providing an interactive interface for real-time predictions.
The key objective is to address the practical challenge of Review Spam Detection  by training a Logistic Regression Classifier on TF-IDF vectorized text data and saving the model for efficient deployment.


## Why This Project Was Built

The presence of fake and bot-generated reviews impacts online shopping experiences, damages product credibility, and undermines customer trust. This project was designed to:

  Detect misleading and bot-generated reviews on e-commerce platforms.
  Explore text classification using machine learning, addressing class imbalance challenges.
  Develop a full-stack ML workflow—from preprocessing to deployment.

## Use Cases

This model has broader applicability across several domains:

  **E-commerce Platforms** – Identifying fake reviews or auto-generated promotional content.
  
  **Social Media Moderation** – Filtering and flagging bot-generated comments.
  
  **Customer Feedback Systems** – Detecting and excluding fraudulent survey responses.
  
  **Email Spam Detection** – Adaptable for filtering out unsolicited or auto-generated emails.
  
  **Research Applications** – Studying the effectiveness of spam classifiers on real-world datasets.


## Technologies Used

  **Python** – For data loading, preprocessing, and model development.
  **Scikit-learn** – For TF-IDF vectorization, model training, and evaluation.
  **Pandas & NumPy** – For structured data manipulation and analysis.
  **Streamlit** – To create a lightweight, user-friendly web interface.
  **Joblib** – To serialize and reuse the trained model and vectorizer.
  **Regular Expressions (Regex)** – For cleaning raw textual data effectively.


## How This Project Is Useful

  Detects spam or bot-generated reviews in real-time.
  Offers confidence scores for every prediction.
  Provides an end-to-end ML pipeline demonstration from data ingestion to user interaction.
  Empowers non-technical users to run predictions via a clean and simple web interface.
