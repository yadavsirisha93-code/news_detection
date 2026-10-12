import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# -----------------------------
# PAGE SETTINGS
# -----------------------------

st.set_page_config(
    page_title="AI News Detection",
    page_icon="📰",
    layout="centered"
)


# -----------------------------
# TRAINING DATA
# -----------------------------

data = {
    "text": [
        "Government announces new education policy for students",
        "Scientists discover a new species in the ocean",
        "India launches a new satellite successfully",
        "Government announces new railway projects",
        "Researchers develop a new medical technology",
        "Weather department predicts heavy rainfall this week",
        "University announces new scholarship program",
        "NASA reports successful space mission",

        "Drinking hot water cures every disease instantly",
        "Scientists confirm humans can live without sleep",
        "Celebrity says eating one fruit gives superpowers",
        "Government gives free cars to every citizen",
        "Doctors reveal that all diseases can be cured in one day",
        "Aliens have officially landed in India",
        "Mobile phone can charge itself without electricity",
        "People can become invisible by drinking special water"
    ],

    "label": [
        "REAL",
        "REAL",
        "REAL",
        "REAL",
        "REAL",
        "REAL",
        "REAL",
        "REAL",

        "FAKE",
        "FAKE",
        "FAKE",
        "FAKE",
        "FAKE",
        "FAKE",
        "FAKE",
        "FAKE"
    ]
}

df = pd.DataFrame(data)


# -----------------------------
# TEXT CLEANING
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)


# -----------------------------
# AI MODEL
# -----------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

model.fit(df["clean_text"], df["label"])


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("📰 AI-Based News Detection System")

st.write(
    "Enter a news article below and the AI model "
    "will classify it as REAL or FAKE."
)

st.markdown("---")


news = st.text_area(
    "📝 Enter News Article",
    height=200,
    placeholder="Type or paste news here..."
)


if st.button("🔍 Detect News", use_container_width=True):

    if not news.strip():

        st.warning("⚠️ Please enter a news article.")

    else:

        cleaned_news = clean_text(news)

        # Prediction
        prediction = model.predict([cleaned_news])[0]

        # Confidence
        probabilities = model.predict_proba([cleaned_news])[0]
        confidence = max(probabilities) * 100

        st.markdown("---")

        if prediction == "REAL":

            st.success("✅ REAL NEWS")

        else:

            st.error("❌ FAKE NEWS")

        st.metric(
            "AI Confidence",
            f"{confidence:.2f}%"
        )


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("About Project")

st.sidebar.write(
    "This project uses Artificial Intelligence "
    "and Machine Learning to classify news."
)

st.sidebar.write("### Technologies")

st.sidebar.write("Python")
st.sidebar.write("Streamlit")
st.sidebar.write("Pandas")
st.sidebar.write("Scikit-learn")
st.sidebar.write("TF-IDF")
st.sidebar.write("Logistic Regression")

st.sidebar.write("### AI Process")

st.sidebar.write(
    "News → Text Cleaning → TF-IDF → "
    "ML Model → Prediction"
)