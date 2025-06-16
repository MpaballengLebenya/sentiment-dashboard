import streamlit as st
from transformers import pipeline
from keybert import KeyBERT
import pandas as pd

# Load models
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
kw_model = KeyBERT()

# Streamlit UI
st.title("🧠 Sentiment Analysis Dashboard")

text = st.text_area("Enter your text:")

if text:
    sentiment = sentiment_pipeline(text)[0]
    keywords = kw_model.extract_keywords(text)

    st.subheader("Results")
    st.write(f"**Sentiment:** {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
    st.write("**Top Keywords:**", [kw[0] for kw in keywords])

