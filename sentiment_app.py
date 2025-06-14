import joblib
import streamlit as st

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

text = st.text_input("Masukkan teks")
if st.button("Prediksi"):
    X = vectorizer.transform([text])
    result = model.predict(X)[0]
    if result == 1:
        st.write(f"Sentimen: Positif")
    else:
        st.write(f"Sentimen: Negatif")
