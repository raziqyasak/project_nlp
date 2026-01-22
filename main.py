import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# Load Model & Vectorizer
# ===============================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

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
