import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import os
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ===============================
# NLTK Setup
# ===============================
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ===============================
# Text Cleaning
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# ===============================
# Train Model
# ===============================
def train_model(fake_file, true_file):
    fake = pd.read_csv(fake_file)
    true = pd.read_csv(true_file)

    fake['label'] = 0
    true['label'] = 1

    # Combine title + text if title exists
    if 'title' in fake.columns and 'title' in true.columns:
        fake['text'] = fake['title'] + " " + fake['text']
        true['text'] = true['title'] + " " + true['text']

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

    accuracy = accuracy_score(y_test, model.predict(X_test))

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer, accuracy, len(fake), len(true)

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title(" Fake News Detection System")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header(" Upload Dataset & Train Model")
fake_file = st.sidebar.file_uploader("Upload Fake News CSV", type=["csv"])
true_file = st.sidebar.file_uploader("Upload True News CSV", type=["csv"])

if st.sidebar.button("Train Model"):
    if fake_file and true_file:
        with st.spinner("Training model..."):
            model, vectorizer, acc, fake_n, true_n = train_model(fake_file, true_file)
        st.sidebar.success(f" Training Completed\nAccuracy: {acc:.2f}")
        st.sidebar.subheader(" Dataset Summary")
        st.sidebar.write(f"Fake News: {fake_n}")
        st.sidebar.write(f"True News: {true_n}")
    else:
        st.sidebar.warning(" Upload both Fake & True datasets")

# -------------------------------
# Load Model
# -------------------------------
if not (os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl")):
    st.warning(" Please train the model first")
    st.stop()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader(" Check News Authenticity")

news_title = st.text_input("News Title")
news_text = st.text_area("News Content", height=200)

# Prediction History
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning(" Please enter news content")
    else:
        full_text = news_title + " " + news_text
        cleaned = clean_text(full_text)

        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        confidence = proba.max() * 100

        # Result
        if prediction == 1:
            st.success(f" REAL News ({confidence:.2f}%)")
        else:
            st.error(f" FAKE News ({confidence:.2f}%)")

        # Confidence Level
        if confidence > 85:
            level = "Very High"
        elif confidence > 70:
            level = "High"
        elif confidence > 55:
            level = "Medium"
        else:
            level = "Low"

        st.info(f" Confidence Level: **{level}**")

        # -------------------------------
        # Pie Chart
        # -------------------------------
        labels = ['FAKE', 'REAL']
        fig, ax = plt.subplots()
        ax.pie(proba, labels=labels, autopct='%1.1f%%')
        ax.set_title("Prediction Probability")
        st.pyplot(fig)

        # -------------------------------
        # Explanation (Keywords)
        # -------------------------------
        st.subheader(" Reason for Prediction")

        feature_names = np.array(vectorizer.get_feature_names_out())
        log_prob = model.feature_log_prob_
        class_idx = prediction

        words = cleaned.split()
        words = [w for w in words if w in feature_names]

        influence = {}
        for w in words:
            idx = np.where(feature_names == w)[0][0]
            influence[w] = log_prob[class_idx][idx]

        top_words = sorted(influence, key=influence.get, reverse=True)[:5]

        if top_words:
            st.write(
                f"This news is predicted as **{'REAL' if prediction==1 else 'FAKE'}** "
                f"because it contains influential keywords such as: **{', '.join(top_words)}**"
            )
        else:
            st.write("No strong keywords detected.")

        # -------------------------------
        # Sensational Language Detection
        # -------------------------------
        sensational_words = [
            "shocking", "unbelievable", "secret", "exposed",
            "breaking", "miracle", "warning", "you won’t believe"
        ]

        detected = [w for w in sensational_words if w in cleaned]
        if detected:
            st.warning(f" Sensational words detected: **{', '.join(detected)}**")

        # -------------------------------
        # Save History
        # -------------------------------
        st.session_state.history.append({
            "Title": news_title,
            "Prediction": "REAL" if prediction == 1 else "FAKE",
            "Confidence (%)": f"{confidence:.2f}"
        })

# -------------------------------
# Prediction History Table
# -------------------------------
if st.session_state.history:
    st.subheader(" Prediction History")
    st.table(pd.DataFrame(st.session_state.history))
