import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

MODEL_DIR = 'models'
DATA_DIR = 'data'
INTENT_FILE = os.path.join(DATA_DIR, 'intents.csv')
FAQ_FILE = os.path.join(DATA_DIR, 'faqs.csv')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

lemmatizer = WordNetLemmatizer()

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    lem = [lemmatizer.lemmatize(t) for t in tokens if t.strip()]
    return " ".join(lem)

# Load full expanded intents file
intents_df = pd.read_csv(INTENT_FILE)
intents_df['question'] = intents_df['question'].fillna("")
intents_df['response'] = intents_df['response'].fillna("")

intents_df['question_norm'] = intents_df['question'].apply(normalize_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
X = vectorizer.fit_transform(intents_df['question_norm'])
y = intents_df['intent']

# Classifier
clf = LogisticRegression(max_iter=1500, solver='saga', multi_class='multinomial')
clf.fit(X, y)

# Save model
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vec.joblib'))
joblib.dump(clf, os.path.join(MODEL_DIR, 'clf_lr.joblib'))

# Generate FAQ file — one canonical Q/A per intent
faq_rows = []
for intent in intents_df['intent'].unique():
    row = intents_df[intents_df['intent'] == intent].iloc[0]
    faq_rows.append({
        'intent': intent,
        'question': row['question'],
        'answer': row['response'],
        'source': ''
    })

pd.DataFrame(faq_rows).to_csv(FAQ_FILE, index=False)
print("Training complete.")
