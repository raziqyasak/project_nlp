import streamlit as st
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# FIX FILE PATH (IMPORTANT!)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Debug: show files in folder (helpful)
st.write("📂 Files in directory:", os.listdir(BASE_DIR))

# ===============================
# Load Model & Vectorizer Safely
# ===============================
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
except Exception as e:
    st.error("❌ Model files not found!")
    st.error(str(e))
    st.stop()

# ===============================
# Text Cleaning
# ===============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is **FAKE** or **REAL**.")

news_text = st.text_area("News Text", height=200)

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max() * 100

        if prediction == 1:
            st.success(f"✅ REAL News ({probability:.2f}%)")
        else:
            st.error(f"❌ FAKE News ({probability:.2f}%)")

st.markdown("---")
st.caption("NLP Individual Project | Fake News Detection")
