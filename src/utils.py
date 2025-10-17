import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Optional

# Download NLTK data (should run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except Exception as e:
    print(f"Error finding corpora/wordnet: {e}, defaulting to download wordnet")
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except Exception as e:
    print(f"Error finding corpora/omw-1.4: {e}, defaulting to download omw-1.4")
    nltk.download('omw-1.4', quiet=True)


# Global variables for model and vectorizer
model: Optional[LogisticRegression] = None
vectorizer: Optional[TfidfVectorizer] = None
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# def preprocess_text(text: str) -> str:
#     """Clean and preprocess text"""
#     text = str(text).lower()
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text)
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     words = text.split()
#     words = [lemmatizer.lemmatize(word) for word in words
#              if word not in stop_words and len(word) > 2]
#     return ' '.join(words)

# def train_model() -> float:
#     """Train the sentiment analysis model"""
#     global model, vectorizer
    
#     print("Training sentiment analysis model...")
    
#     # Load data or create sample data
#     try:
#         df = pd.read_csv('amazonreviews.tsv', sep='\t', encoding='utf-8')
#     except:
#         print("Creating sample dataset...")
#         sample_data = {
#             'review': ["This product is amazing!", "Terrible quality", "It's okay"] * 100,
#             'label': ['pos', 'neg', 'pos'] * 100
#         }
#         df = pd.DataFrame(sample_data)
    
#     df['clean_review'] = df['review'].apply(preprocess_text)
#     df = df[df['clean_review'].str.len() > 0]
    
#     vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
#     X = vectorizer.fit_transform(df['clean_review'])
#     y = df['label']
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     model.fit(X_train, y_train)
    
#     accuracy = model.score(X_test, y_test)
#     print(f"Model trained with accuracy: {accuracy:.2f}")
    
#     # Save model and vectorizer
#     with open('sentiment_model.pkl', 'wb') as f:
#         pickle.dump(model, f)
#     with open('vectorizer.pkl', 'wb') as f:
#         pickle.dump(vectorizer, f)
    
#     return accuracy




lemmatizer = WordNetLemmatizer()
# We keep the standard stop word set
stop_words = set(stopwords.words('english'))
# Define common negation terms
NEGATION_WORDS = ['not', 'no', 'never', 'n\'t']


def preprocess_text(text):
    """Clean, handle negation, and preprocess text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 1. Tokenize (keeping apostrophes for "n't" to be handled correctly)
    words = re.findall(r'\b\w+\b|\S', text) # Simple tokenization
    
    # 2. NEGATION HANDLING LOOP
    new_words = []
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check if the word is a negation word AND there is a word immediately following it
        if word in NEGATION_WORDS and i + 1 < len(words):
            next_word = words[i+1]
            
            # Combine them: 'not' + 'good' -> 'not_good'
            new_words.append(f"not_{next_word}")
            i += 2  # Skip the next word since it's now part of the negation tag
        
        # Handle cases like "didn't" by replacing with "not_" and skipping the "n't"
        elif word.endswith("n't"):
            # If "didn't", we'll just process it as a single token but it won't be in stop_words
            new_words.append(word)
            i += 1
            
        # Standard words
        else:
            new_words.append(word)
            i += 1
            
    # 3. Final Cleaning, Lemmatization, and Stop Word Removal
    # Join and clean again to remove any stray punctuation/symbols
    final_text = ' '.join(new_words)
    final_text = re.sub(r'[^a-zA-Z_]', ' ', final_text).strip() # only keep letters and underscores
    final_words = final_text.split()
    
    processed_words = [
        lemmatizer.lemmatize(word) for word in final_words 
        if word not in stop_words and len(word) > 2
    ]
    
    return ' '.join(processed_words)


import joblib 

def load_and_predict(text_to_analyze: str):
    """Loads models using joblib and predicts sentiment for a given text."""
    model = None
    vectorizer = None
    
    # 1. Load the vectorizer and classifier using JOBLIB
    try:
        # Changed pickle.load to joblib.load
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = joblib.load(f)
        with open('models/classifier.pkl', 'rb') as f:
            model = joblib.load(f)
        print("✓ Models loaded successfully using joblib.")
    except FileNotFoundError:
        return {"error": "Model files not found. Check the 'models/' path."}
    except Exception as e:
        # This will now catch joblib-related errors if they occur
        return {"error": f"Failed to load models with joblib: {e}"}

    # ... (rest of the prediction logic remains the same)
    # 2. Preprocess the input text
    cleaned_text = preprocess_text(text_to_analyze)
    
    if not cleaned_text:
        return {"sentiment": "neutral", "confidence": 0.0, "text": text_to_analyze}

    # 3. Vectorize the text using the LOADED vectorizer
    features = vectorizer.transform([cleaned_text])

    # 4. Predict sentiment and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Simple output format
    return {
        "text": text_to_analyze,
        "sentiment": prediction,
        "confidence": max(probabilities),
        "probabilities": {
            "positive": probabilities[1],
            "negative": probabilities[0]
        }
    }

class ModelStore:
    model = None
    vectorizer = None

def load_model() -> bool:
    """Load saved model and vectorizer, or train if not found"""
    if ModelStore.model and ModelStore.vectorizer:
        print("Model already loaded.")
        return True
    
    if ModelStore.model is not None and ModelStore.vectorizer is not None:
        print("Model already loaded.")
        return True

    try:
        # Changed pickle.load to joblib.load
        with open('models/vectorizer.pkl', 'rb') as f:
            ModelStore.vectorizer = joblib.load(f)
        with open('models/classifier.pkl', 'rb') as f:
            ModelStore.model = joblib.load(f)
        print("✓ Models loaded successfully using joblib.")
        return True
    except FileNotFoundError:
        print("Model files not found. Check the 'models/' path.")
        return False
    except Exception as e:
        # This will now catch joblib-related errors if they occur
        print(f"Failed to load models with joblib: {e}")
        return False

def predict_sentiment(text: str) -> Dict[str, Any]:
    """Predict sentiment with confidence scores"""
    global model, vectorizer
    if model is None or vectorizer is None:
        raise RuntimeError("Model or vectorizer not loaded. Run load_model() first.")


    cleaned = preprocess_text(text)
    if not cleaned:
         return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'probabilities': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        }
    
    features = vectorizer.transform([cleaned])
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Determine the index for positive and negative
    try:
        pos_idx = list(model.classes_).index('pos')
        neg_idx = list(model.classes_).index('neg')
    except ValueError:
        # Fallback if classes are not 'pos'/'neg' (shouldn't happen with our training)
        pos_idx, neg_idx = 1, 0 
        
    prob_pos = float(probabilities[pos_idx])
    prob_neg = float(probabilities[neg_idx])
    max_prob = max(prob_pos, prob_neg)
    
    # Logic for neutral
    if max_prob < 0.6:
        sentiment = 'neutral'
        confidence = 1 - max_prob
        prob_neutral = confidence
    else:
        sentiment = 'positive' if prediction == 'pos' else 'negative'
        confidence = max_prob
        prob_neutral = 1.0 - max_prob
    
    return {
        'sentiment': sentiment,
        'confidence': float(confidence),
        'probabilities': {
            'positive': prob_pos,
            'negative': prob_neg,
            'neutral': prob_neutral
        }
    }

def get_model_status() -> Dict[str, bool]:
    return {
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    }

def validate_review_text(text: str) -> (bool, str):
    """Validate review text input"""
    if not isinstance(text, str):
        return False, "Review text must be a string."
    if len(text.strip()) == 0:
        return False, "Review text cannot be empty."
    if len(text) > 1000:
        return False, "Review text exceeds maximum length of 1000 characters."
    return True, ""  #validate_review_text, clean_review_batch


def clean_review_batch(reviews: List[str]) -> List[str]:
    """Clean and validate a batch of reviews"""
    cleaned_reviews = []
    for review in reviews:
        is_valid, error = validate_review_text(review)
        if is_valid:
            cleaned_reviews.append(review)
    return cleaned_reviews


def load_sample_reviews() -> List[str]:
    """Load sample reviews from a local file or create sample data"""
    try:
        df = pd.read_csv('amazonreviews.tsv', sep='\t', encoding='utf-8')
        return df['review'].dropna().tolist()
    except Exception as e:
        print(f"Error loading sample reviews: {e}")
        # Return some hardcoded samples if file not found
        return [
            "This product is amazing! I loved it.",
            "Terrible quality, very disappointed.",
            "It's okay, does the job but nothing special.",
            "Exceeded my expectations, highly recommend!",
            "Not worth the price, would not buy again."
        ]
