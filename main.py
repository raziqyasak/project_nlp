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

nltk.download('stopwords')
nltk.download('wordnet')

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
# Train Model Function
# ===============================
def train_model(fake_file, true_file):
    fake = pd.read_csv(fake_file)
    true = pd.read_csv(true_file)

    fake['label'] = 0
    true['label'] = 1

    # Combine title + content if title exists
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

    acc = accuracy_score(y_test, model.predict(X_test))

    # Save model & vectorizer
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer, acc

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection System")

# -------------------------------
# Sidebar: Upload & Train
# -------------------------------
st.sidebar.header("⚙️ Upload Dataset and Train Model")
fake_file = st.sidebar.file_uploader("Upload Fake News CSV", type=["csv"])
true_file = st.sidebar.file_uploader("Upload True News CSV", type=["csv"])

if st.sidebar.button("Train Model"):
    if fake_file and true_file:
        with st.spinner("Training model..."):
            model, vectorizer, accuracy = train_model(fake_file, true_file)
        st.sidebar.success(f"✅ Model trained!\nAccuracy: {accuracy:.2f}")
    else:
        st.sidebar.warning("⚠️ Please upload both Fake and True CSV files.")

# -------------------------------
# Load Model if exists
# -------------------------------
if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
else:
    st.warning("⚠️ Model not trained yet. Please upload dataset and train using sidebar.")
    st.stop()

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("Enter a news article to check:")
news_title = st.text_input("News Title")
news_text = st.text_area("News Content", height=200)

if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some content.")
    else:
        # Combine title + content
        full_text = news_title + " " + news_text
        cleaned = clean_text(full_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]
        probability = proba.max() * 100

        # Display prediction
        if prediction == 1:
            st.success(f"✅ REAL News ({probability:.2f}%)")
        else:
            st.error(f"❌ FAKE News ({probability:.2f}%)")

        # -------------------------------
        # Pie Chart for probability
        # -------------------------------
        labels = ['FAKE', 'REAL']
        sizes = proba
        colors = ['red', 'green']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax.set_title("Prediction Probability")
        st.pyplot(fig)

        # -------------------------------
        # Reason / Explanation (dynamic)
        # -------------------------------
        st.subheader("Reason for Prediction:")

        feature_names = np.array(vectorizer.get_feature_names_out())
        log_prob = model.feature_log_prob_
        class_idx = prediction  # 0 = Fake, 1 = Real

        words_in_text = cleaned.split()
        words_in_vocab = [w for w in words_in_text if w in feature_names]

        word_influence = {}
        for w in words_in_vocab:
            idx = np.where(feature_names == w)[0][0]
            word_influence[w] = log_prob[class_idx][idx]

        top_words = sorted(word_influence, key=word_influence.get, reverse=True)[:5]

        if len(top_words) > 0:
            st.write(
                f"This news is predicted as **{'REAL' if prediction==1 else 'FAKE'}** because it contains keywords like: **{', '.join(top_words)}**"
            )
        else:
            st.write("No strong keywords found in this text to explain prediction.")
