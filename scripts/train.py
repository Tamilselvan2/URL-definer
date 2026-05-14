#!/usr/bin/env python3
"""
Offline training script for URL classifier.
Run this to train/retrain the model independently from the Flask app.

Usage:
    python scripts/train.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, classification_report
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model and feature extractor paths
MODEL_PATH = 'url_classifier.pkl'
FEATURE_EXTRACTOR_PATH = 'feature_extractor.pkl'
DATASET_PATH = 'dataset.csv'

# Constants
SUSPICIOUS_KEYWORDS = [
    'login', 'secure', 'verify', 'free', 'win', 'promo', 'gift', 'click',
    'update', 'account', 'confirm', 'deal', 'redeem', 'prize', 'join'
]


def extract_handcrafted_features(urls):
    """Extract handcrafted features from URLs."""
    features = []
    for url in urls:
        try:
            parsed = urlparse(url)
            from urllib.parse import parse_qs
            query_params = parse_qs(parsed.query)
            
            url_length = len(url)
            num_query_params = len(query_params)
            has_suspicious = int(any(keyword in url.lower() for keyword in SUSPICIOUS_KEYWORDS))
            domain = parsed.netloc.lower().replace('www.', '')
            num_subdomains = len(domain.split('.')) - 2 if domain else 0
            is_https = int(parsed.scheme == 'https')
            
            features.append([url_length, num_query_params, has_suspicious, num_subdomains, is_https])
        except Exception as e:
            logger.warning(f"Error extracting features for URL '{url}': {str(e)}")
            features.append([0, 0, 0, 0, 0])
    
    return np.array(features)


def train_model():
    """Train the SVM classifier and save artifacts."""
    try:
        # Load dataset
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset not found at {DATASET_PATH}")
            logger.info("Please ensure dataset.csv exists in the project root.")
            return False
        
        logger.info(f"Loading dataset from {DATASET_PATH}")
        data = pd.read_csv(DATASET_PATH)
        
        if len(data) == 0:
            logger.error("Dataset is empty. Cannot train model.")
            return False
        
        logger.info(f"Dataset loaded with {len(data)} rows")
        
        # Prepare data
        data['label'] = data['label'].str.capitalize()
        X = data['url']
        y = data['label']
        
        logger.info(f"Label distribution:\n{y.value_counts()}")
        
        # Create feature extractor
        logger.info("Creating feature union (TF-IDF + handcrafted features)...")
        feature_extractor = FeatureUnion([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                token_pattern=r'(?u)\b\w+\b|[^\w\s]',
                max_features=5000
            )),
            ('handcrafted', FunctionTransformer(extract_handcrafted_features, validate=False))
        ])
        
        # Create classifier with probability=True for calibrated probabilities
        classifier = Pipeline([
            ('features', feature_extractor),
            ('clf', SVC(kernel='linear', random_state=42, probability=True))
        ])
        
        # Train model
        logger.info("Training SVM classifier...")
        classifier.fit(X, y)
        logger.info("Model trained successfully")
        
        # Cross-validation
        logger.info("Running 5-fold cross-validation...")
        cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='f1_macro')
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Average F1 score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Save artifacts
        logger.info(f"Saving classifier to {MODEL_PATH}")
        joblib.dump(classifier, MODEL_PATH)
        
        logger.info(f"Saving feature extractor to {FEATURE_EXTRACTOR_PATH}")
        joblib.dump(feature_extractor, FEATURE_EXTRACTOR_PATH)
        
        logger.info("✓ Model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("URL Classifier Model Training")
    logger.info("=" * 60)
    
    success = train_model()
    
    logger.info("=" * 60)
    if success:
        logger.info("Training completed successfully.")
        sys.exit(0)
    else:
        logger.error("Training failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
