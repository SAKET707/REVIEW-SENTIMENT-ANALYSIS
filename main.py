import streamlit as st
from prediction_helper import predict

st.title("Review Sentiment Predictor")
review = st.text_input("Enter your Review:", "")
sentiment_map = ["Negative","Neutral","Positive"]

if st.button("Predict"):
    if not review.strip():
        st.error("Review cannot be empty!")
    else:
        pred = predict(review)
        st.succes(pred-1)
        st.success(f"Predicted Sentiment: {sentiment_map[pred]}")