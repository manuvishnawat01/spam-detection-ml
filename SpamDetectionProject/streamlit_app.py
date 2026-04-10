import streamlit as st
import pandas as pd
import os

# Load dataset (SAFE PATH)
data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "spam.csv"),
    sep='\t',
    names=["label", "message"]
)

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X = data['message']
y = data['label']

# Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(X)

# Model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.5)
model.fit(X, y)

# UI
st.title("📩 Spam Detection App")
st.write("Enter a message to check if it's Spam or Not Spam")

msg = st.text_area("Enter message")

if st.button("Check"):
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    probability = model.predict_proba(msg_vector)[0]

    spam_prob = probability[1] * 100
    not_spam_prob = probability[0] * 100

    if prediction == 1:
        st.error(f"Spam ❌ ({spam_prob:.2f}%)")
    else:
        st.success(f"Not Spam ✅ ({not_spam_prob:.2f}%)")