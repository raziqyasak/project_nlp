import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# File Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAKE_PATH = os.path.join(BASE_DIR, "Fake-news.csv")
TRUE_PATH = os.path.join(BASE_DIR, "True-news.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# ===============================
# Text Cleaning
# ===============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ===============================
# Train Model (ONLY ONCE)
# ===============================
def train_model():
    fake = pd.read_csv(FAKE_PATH)
    true = pd.read_csv(TRUE_PATH)

    fake['label'] = 0
    true['label'] = 1

    df = pd.concat([fake, true])
    df = df[['text', 'label']].dropna()

    df['clean_text'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

    return acc

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")

# ===============================
# Sidebar: Model Training
# ===============================
st.sidebar.title("⚙️ Model Control")

if st.sidebar.button("Train Model Using Dataset"):
    with st.spinner("Training model..."):
        accuracy = train_model()
    st.sidebar.success(f"✅ Model trained successfully!\nAccuracy: {accuracy:.2f}")

# ===============================
# Load Model
# ===============================
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
else:
    st.warning("⚠️ Model not trained yet. Please train using sidebar.")
    st.stop()

# ===============================
# Prediction UI
# ===============================
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
