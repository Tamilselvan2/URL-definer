import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv('dataset.csv')

# Extract features and labels
X = data['url']
y = data['label']

# Vectorize URLs
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
