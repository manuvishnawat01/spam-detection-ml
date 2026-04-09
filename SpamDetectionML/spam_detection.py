import pandas as pd

# Load dataset
data = pd.read_csv("spam.csv", sep='\t', names=["label", "message"])

# Convert labels into numbers (spam = 1, ham = 0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

extra_data = pd.DataFrame({
    'label': [1]*30,
    'message': [
        "free cash prize waiting for you",
        "you are a lucky winner claim your reward",
        "win money now click here",
        "urgent money required act now",
        "lottery winner claim your jackpot",
        "bonus reward waiting claim now",
        "last chance to win big prize",
        "offer expires soon hurry up",
        "special offer just for you",
        "get huge discount on products",
        "cheap deals available buy now",
        "exclusive deal click here now",
        "buy now and get bonus reward",
        "claim your lottery prize now",
        "jackpot winner announcement claim now",
        "urgent response needed to claim reward",
        "free entry to win cash prize",
        "limited time offer buy now",
        "click here to claim your bonus",
        "deal of the day buy now",
        "cheap price limited stock hurry",
        "exclusive discount offer expires soon",
        "winner you got a free reward",
        "claim now before offer expires",
        "urgent lottery notification claim prize",
        "win jackpot now click here",
        "bonus cash reward waiting for you",
        "free gift claim now hurry",
        "last chance exclusive deal",
        "special discount buy now"
    ]
})

data = pd.concat([data, extra_data], ignore_index=True)

# Features and target
X = data['message']
y = data['label']

# 🔥 Improved Text Vectorization (TF-IDF + n-grams + filtering)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(X)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Improved Model (tuned Naive Bayes)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy in percentage
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")

# -------- USER INPUT --------
msg = input("\nEnter your message: ")

msg_vector = vectorizer.transform([msg])

# Prediction
prediction = model.predict(msg_vector)[0]

# Probability
probability = model.predict_proba(msg_vector)[0]

spam_prob = probability[1] * 100
not_spam_prob = probability[0] * 100

print("\nResult:")
if prediction == 1:
    print("Spam ❌")
else:
    print("Not Spam ✅")

print(f"Spam Probability: {spam_prob:.2f}%")
print(f"Not Spam Probability: {not_spam_prob:.2f}%")
