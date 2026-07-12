import streamlit as st
import numpy as np
import joblib
import os

# Since all your files are in the same folder, we can import them directly
from preprocess import clean_tweet
from vectorization import MeanEmbeddingVectorizer

st.set_page_config(page_title="Semantic Sentiment Engine", page_icon="💡", layout="centered")

@st.cache_resource
def bootstrap_models():
    """ Load binary pipelines efficiently using Streamlit's resource cacher. """
    vectorizer = MeanEmbeddingVectorizer.load("w2v_model.model")
    classifier = joblib.load("sentiment_classifier.pkl")
    return vectorizer, classifier

st.title("💡 Sentiment Analysis Engine via Word2Vec")
st.markdown("""
This production application converts unstructured syntax configurations into dense continuous vector embeddings 
to gauge downstream user expressions.
""")

try:
    vectorizer, classifier = bootstrap_models()
    st.sidebar.success("✅ Semantic Core Loaded")
except Exception as e:
    st.error(f"Execution Error: {e}. Ensure you have run train.py first to generate the models/ folder.")
    st.stop()

st.write("---")
user_input = st.text_area("Analyze Text / Tweet Sentiments Realtime:", placeholder="Type review insights here...")

if st.button("Evaluate Sentiment Matrix", type="primary"):
    if not user_input.strip():
        st.warning("Input requires syntax entities to classify.")
    else:
        # Preprocessing & Vectorization 
        tokens = clean_tweet(user_input)
        
        if not tokens:
            st.error("Text structures lacked valid semantic identifiers. Try alternative configurations.")
        else:
            doc_vector = vectorizer.transform([tokens])
            
            # Predict
            prediction = classifier.predict(doc_vector)[0]
            probabilities = classifier.predict_proba(doc_vector)[0]
            target_labels = classifier.classes_
            
            # Interface Visual Styling Maps
            color_scheme = {"positive": "green", "neutral": "orange", "negative": "red"}
            color = color_scheme.get(prediction.lower(), "blue")
            
            st.markdown(f"### Classification Prediction: :{color}[{prediction.upper()}]")
            
            # Metric Probability Distributions
            st.write("#### Confidence Distribution Metric Matrix:")
            for label, prob in zip(target_labels, probabilities):
                st.write(f"- **{label.capitalize()} Sentiment Confidence**: {prob * 100:.2f}%")
                st.progress(float(prob))
