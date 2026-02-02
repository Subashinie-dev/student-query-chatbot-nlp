# Student Query Support Chatbot (NLP)

This project is an NLP-based chatbot designed to provide automated responses to common academic and administrative student queries.

## Problem Statement
Student support in colleges is often handled manually through helpdesks, emails, or in-person visits, leading to delays and increased workload. Many existing chatbot solutions rely on heavy deep learning models that require large datasets and high computational resources.

## Proposed Solution
A lightweight chatbot that uses traditional NLP and machine learning techniques to efficiently classify user intent and retrieve relevant responses with low latency.

## Key Features
- Text preprocessing using NLP techniques
- TF-IDF vectorization for feature extraction
- Logistic Regression for intent classification
- Cosine similarity for matching user queries with FAQs
- Domain-specific synonym expansion for better understanding
- Multi-intent handling with fallback responses
- Flask-based web interface for real-time interaction

## Technologies Used
- Python
- Flask
- Scikit-learn
- NLTK
- Pandas, NumPy
- HTML, CSS, JavaScript

## Advantages
- Fast response time (< 1 second)
- Works well with small datasets
- Low computational and hardware requirements
- Easy to update and maintain
- Suitable for educational institutions

## Future Enhancements
- Integration with advanced NLP models (BERT/GPT)
- Voice-based query support
- Multi-language support
- Integration with college databases
- Deployment on mobile and messaging platforms

## Project Status
Academic project – open to further improvements
