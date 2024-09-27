import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("complaints.csv")


def preprocess_text(text):
    return text


data["complaint_text"] = data["complaint_text"].apply(preprocess_text)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["complaint_text"])
y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)


def predict(text):
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text])
    predicted_category = model.predict(vectorized_text)[0]
    return predicted_category

print("Enter your problem")
new_complaint = input()
predicted_category = predict(new_complaint)
print(predicted_category)