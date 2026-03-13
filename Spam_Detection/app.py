import streamlit as st
import joblib, re, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))
model      = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ── UI ──
st.set_page_config(page_title="Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.markdown("Enter any SMS message to check if it's **spam** or **ham**.")

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''

st.sidebar.header("💡 Try These Examples")
examples = [
    "FREE MONEY! Click now to claim your prize!!!",
    "Hey, are you coming to the meeting tomorrow?",
    "URGENT! You have won £1000. Claim now!",
    "Ok lar, joking wif u oni...",
    "Had your mobile 11 months or more? U R entitled to update!"
]
for ex in examples:
    if st.sidebar.button(ex[:45] + "..."):
        st.session_state['input_text'] = ex

text = st.text_area("✉️ SMS Message",
                     value=st.session_state['input_text'],
                     placeholder="Type your message here...",
                     height=150,
                     key="input_text")

if st.button("🔍 Classify"):
    if text.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        cleaned    = preprocess(text)
        vectorized = vectorizer.transform([cleaned])
        pred       = model.predict(vectorized)[0]
        label      = "spam" if pred == 1 else "ham"

        if label == 'spam':
            st.error("🚨 SPAM detected!")
        else:
            st.success("✅ HAM — Looks legit!")

        with st.expander("🔎 See details"):
            st.write(f"**Original:**   {text}")
            st.write(f"**Cleaned:**    {cleaned}")
            st.write(f"**Prediction:** {label.upper()}")
