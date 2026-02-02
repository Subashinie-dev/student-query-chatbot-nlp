# app.py (modified)

import os
import uuid
import json
from flask import Flask, request, jsonify, render_template, session
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
import numpy as np
import datetime
import csv
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-random-secret'

MODEL_DIR = 'models'
VECT_PATH = os.path.join(MODEL_DIR, 'tfidf_vec.joblib')
CLF_PATH  = os.path.join(MODEL_DIR, 'clf_lr.joblib')
FAQ_PATH  = 'data/faqs.csv'

lemmatizer = WordNetLemmatizer()

def normalize_text(text):
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Manual synonym map (domain-specific)
SYN_MAP = {
    'admission': ['admission','apply','application','enroll','join','registration'],
    'fees': ['fee','tuition','charges','payment','bill'],
    'hostel': ['hostel','room','accommodation','dorm'],
    'mess': ['mess','food','canteen','menu','dining'],
    'exam': ['exam','test','semester','assessment','midterm','lecture'],
    'attendance': ['attendance','present','absent','percentage','attendance rule'],
    'events': ['event','fest','workshop','seminar'],
    'scholarship': ['scholarship','grant','fellowship','financial aid'],
    'library': ['library','books','reading'],
    'transport': ['transport','bus','route','pass'],
    'ragging': ['ragging','harassment','bully','complaint'],
    'contact': ['contact','phone','email','helpdesk','support'],
    'general': ['general','info','information','campus','college details']
}

WORD_TO_KEYS = {}
for k,lst in SYN_MAP.items():
    for w in lst:
        WORD_TO_KEYS[w] = k

# Load FAQ KB
def load_faqs():
    if not os.path.exists(FAQ_PATH):
        df = pd.DataFrame([
            {'intent':'admission','question':'what is admission start date','answer':'Admissions start on July 1, 2025. Check admissions page.','source':''},
            {'intent':'fees','question':'what is fee for BCA','answer':'BCA annual fee is ₹40,000. Check fees page.','source':''},
            {'intent':'hostel','question':'hostel availability','answer':'Hostel rooms allotted based on merit and availability.','source':''},
        ])
        df.to_csv(FAQ_PATH,index=False)
    return pd.read_csv(FAQ_PATH)

faqs_df = load_faqs()
faqs_df = faqs_df.fillna("")

# Load models
vectorizer = joblib.load(VECT_PATH)
clf = joblib.load(CLF_PATH)

# Modified synonym expansion: ONLY manual domain-specific synonyms
def expand_synonyms(text):
    tokens = [t for t in word_tokenize(text) if t.strip()]
    expanded = set(tokens)
    for t in tokens:
        key = WORD_TO_KEYS.get(t)
        if key:
            expanded.update(SYN_MAP[key])
            expanded.add(key)
    return " ".join(sorted(expanded))

# Multi-Intent + Fallback settings
ALPHA = 0.6                  # Slightly lower → more weight to semantic similarity
COMBINED_THRESHOLD = 0.25     # Lower threshold → allow multiple low-confidence intents
MAX_MULTI = 3                 # top N intents

@app.route('/')
def index():
    return render_template('index.html')

def safe(val):
    if val is None or (isinstance(val,float) and math.isnan(val)):
        return ""
    return str(val)

@app.route('/api/query', methods=['POST'])
def api_query():
    payload = request.json or {}
    text = payload.get('text','').strip()
    profile = payload.get('profile','').strip()

    if not text:
        return jsonify({'error':'empty query'}), 400

    norm = normalize_text(text)
    expanded = expand_synonyms(norm)

    if 'context' not in session:
        session['context'] = []
    context = session['context']

    X = vectorizer.transform([expanded])
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_

    # Semantic similarity to FAQ questions
    kb_questions = faqs_df['question'].apply(normalize_text).tolist()
    kb_vecs = vectorizer.transform(kb_questions)
    sims = cosine_similarity(X, kb_vecs)[0]

    # Combined scoring
    combined_scores = []
    for cls, prob in zip(classes, probs):
        # check similarity with FAQ questions for this intent
        idxs = faqs_df[faqs_df['intent']==cls].index.tolist()
        sim_val = max([float(sims[i]) for i in idxs]) if idxs else 0.0
        combined = ALPHA*float(prob) + (1-ALPHA)*sim_val
        combined_scores.append({'intent': cls, 
                                'prob': float(prob), 
                                'sim': sim_val, 
                                'combined': combined})

    combined_scores = sorted(combined_scores, key=lambda x: x['combined'], reverse=True)
    picks = [c for c in combined_scores if c['combined']>=COMBINED_THRESHOLD][:MAX_MULTI]

    response_payload = {'query_id': str(uuid.uuid4())}
    answers = []

    if not picks:
        # fallback: best FAQ match
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        if best_score >= 0.15:
            row = faqs_df.iloc[best_idx]
            response_payload['answer'] = row['answer']
            response_payload['intents'] = [{'intent': row['intent'], 'conf': best_score}]
            response_payload['multi'] = [{'answer': row['answer'],'source':safe(row.get('source'))}]
        else:
            response_payload['answer'] = "I'm not fully sure about that. Try asking about admissions," \
            " fees, hostel, exams, placement, transport, or events."
            response_payload['intents'] = [{'intent': combined_scores[0]['intent'], 'conf': combined_scores[0]['prob']}]
            response_payload['multi'] = []
    else:
        for pick in picks:
            intent = pick['intent']
            candidates = faqs_df[faqs_df['intent']==intent]
            if not candidates.empty:
                cand_vecs = vectorizer.transform(candidates['question'].apply(normalize_text).tolist())
                sim_vals = cosine_similarity(X, cand_vecs)[0]
                best = int(sim_vals.argmax())
                answer = candidates.iloc[best]['answer']
                source = safe(candidates.iloc[best].get('source'))
            else:
                answer = f'I found intent {intent} but no prepared answer.'
                source = ''
            answers.append({'intent': intent, 'answer': answer, 'source': source, 'score': pick['combined'], 'prob': pick['prob'], 'sim': pick['sim']})

        response_payload['answer'] = answers[0]['answer']
        response_payload['intents'] = [{'intent':a['intent'],'conf':a['score']} for a in answers]
        response_payload['multi'] = [{'answer':a['answer'],'source':a['source']} for a in answers]

    # personalization
    response_payload['personalized'] = f"Note: answered using profile '{profile}'" if profile else ""

    # short-term session memory
    context.append({'time': datetime.datetime.utcnow().isoformat(), 'query': text, 'top_intent': response_payload['intents'][0]['intent']})
    session['context'] = context[-3:]

    return jsonify(response_payload)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
