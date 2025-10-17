"""
ml_models.py - Machine Learning Model Management
"""

import pickle
import os
import logging
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import settings
from src.utils import preprocess_text

logger = logging.getLogger(__name__)
from joblib import load

# model_path = "lr_models/sentiment_model.pkl"
# vectorizer_path = "lr_models/vectorizer.pkl"
model_path = "models/classifier.pkl"
vectorizer_path = "models/vectorizer.pkl"


def load_model():
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        return model, vectorizer
    return None, None


class SentimentAnalyzer:
    """Singleton class for managing sentiment analysis model"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.model = None
            self.vectorizer = None
            self.is_trained = False
            self.model_metrics = {}
            self.initialized = True
    
    def load_model(self) -> bool:
        """
        Load saved model and vectorizer from disk
        
        Returns:
            Success status
        """
        # try:
        #     if os.path.exists(settings.model_path) and os.path.exists(settings.vectorizer_path):
        #         with open(settings.model_path, 'rb') as f:
        #             self.model = pickle.load(f)
        #         with open(settings.vectorizer_path, 'rb') as f:
        #             self.vectorizer = pickle.load(f)
        #         self.is_trained = True
        #         logger.info("Model and vectorizer loaded successfully")
        #         return True
        #     else:
        #         logger.warning("Model files not found, training required")
        #         return False
        # except Exception as e:
        #     logger.error(f"Error loading model: {e}")
        #     return False
        
        try:
            if os.path.exists(settings.model_path) and os.path.exists(settings.vectorizer_path):
                with open(settings.vectorizer_path, 'rb') as f:
                    self.vectorizer = load(f)  # Changed pickle.load to joblib.load
                with open(settings.model_path, 'rb') as f:
                    self.model = load(f)  # Changed pickle.load to joblib.load
                self.is_trained = True
                logger.info("Model and vectorizer loaded successfully using joblib.")
                return True
            print("✓ Models loaded successfully using joblib.")
        except FileNotFoundError:
            print("✗ Model files not found. Check the 'models/' path.")
            return False
        except Exception as e:
            logger.error(f"Error loading model with joblib: {e}")
            # This will now catch joblib-related errors if they occur
            return False

    
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text
        
        Args:
            text: Review text
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess text
        cleaned = preprocess_text(text)
        
        if not cleaned:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'probabilities': {
                    'positive': 0.33,
                    'negative': 0.33,
                    'neutral': 0.34
                }
            }
        
        # Vectorize
        features = self.vectorizer.transform([cleaned])
        
        # Get prediction and probabilities
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get class indices
        classes = self.model.classes_
        pos_idx = np.where(classes == 'pos')[0][0]
        neg_idx = np.where(classes == 'neg')[0][0]
        
        # Calculate probabilities
        pos_prob = float(probabilities[pos_idx])
        neg_prob = float(probabilities[neg_idx])
        max_prob = max(pos_prob, neg_prob)
        
        # Determine sentiment with confidence threshold
        if max_prob < settings.confidence_threshold:
            sentiment = 'neutral'
            confidence = 1 - max_prob
            neutral_prob = 1 - max_prob
        else:
            sentiment = 'positive' if prediction == 'pos' else 'negative'
            confidence = max_prob
            neutral_prob = 0.0
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': {
                'positive': pos_prob,
                'negative': neg_prob,
                'neutral': neutral_prob
            }
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of review texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def _load_or_create_dataset(self) -> pd.DataFrame:
        """
        Load dataset from file or create sample dataset
        
        Returns:
            DataFrame with review and label columns
        """
        # Try to load from file
        if settings.dataset_path and os.path.exists(settings.dataset_path):
            try:
                df = pd.read_csv(settings.dataset_path, sep='\t', encoding='utf-8')
                logger.info(f"Dataset loaded from {settings.dataset_path}")
                return df
            except Exception as e:
                logger.warning(f"Could not load dataset: {e}")
        
        # Create sample dataset
        logger.info("Creating sample dataset...")
        sample_data = {
            'review': [
                # Positive reviews
                "This product is absolutely amazing! Best purchase ever!",
                "Excellent quality and fast shipping. Highly recommend!",
                "Outstanding product, exceeded all my expectations!",
                "Perfect! Exactly what I was looking for. Great value!",
                "Fantastic quality! Will definitely buy again!",
                "Love this product! Works perfectly and looks great!",
                "Incredible value for money. Very satisfied!",
                "Best purchase I've made this year. Highly recommended!",
                "Superb quality and excellent customer service!",
                "Amazing product! Couldn't be happier with my purchase!",
                
                # Negative reviews
                "Terrible quality, complete waste of money",
                "Product broke after one day. Very disappointed.",
                "Horrible experience. Would not recommend to anyone.",
                "Worst purchase ever. Stay away from this product!",
                "Poor quality materials. Not worth the price.",
                "Defective product. Customer service was unhelpful.",
                "Complete garbage. Threw it away immediately.",
                "Misleading description. Product is nothing like advertised.",
                "Broken on arrival. Very frustrating experience.",
                "Cheap quality. Falls apart easily.",
                
                # Mixed/Neutral reviews
                "It's okay, nothing special but does the job",
                "Average product. Some good features, some bad.",
                "Decent for the price. Not great, not terrible.",
                "Works as expected. Nothing extraordinary.",
                "Acceptable quality. Meets basic requirements."
            ] * 10,  # Repeat for more samples
            'label': (
                ['pos'] * 10 +  # Positive labels
                ['neg'] * 10 +  # Negative labels
                ['pos', 'neg', 'pos', 'pos', 'pos']  # Mixed (will help create neutral threshold)
            ) * 10
        }
        
        return pd.DataFrame(sample_data)
    
    def get_model_info(self) -> Dict:
        """
        Get current model information
        
        Returns:
            Dictionary with model info
        """
        return {
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__ if self.model else None,
            'vectorizer_type': type(self.vectorizer).__name__ if self.vectorizer else None,
            'metrics': self.model_metrics,
            'model_path': settings.model_path,
            'vectorizer_path': settings.vectorizer_path
        }

# Create global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()