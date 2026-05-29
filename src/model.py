# ==========================================
# Author: Ansh Jaiswal
# Neutra-Mod Toxicity Classifier Architectures (Classical & DL)
# ==========================================
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

class ToxicityClassifier:
    def __init__(self, categories=None):
        if categories is None:
            self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        else:
            self.categories = categories
            
        # Default thresholds optimized for high precision
        # Precision > 80% for severe_toxic is key!
        self.thresholds = {
            'toxic': 0.4,
            'severe_toxic': 0.7,  # Raised threshold to ensure > 80% precision
            'obscene': 0.4,
            'threat': 0.5,
            'insult': 0.4,
            'identity_hate': 0.5
        }
        
        # Build classical Pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=25000,
                min_df=3,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('clf', MultiOutputClassifier(LogisticRegression(
                C=2.0,
                class_weight='balanced',
                max_iter=1000,
                n_jobs=-1,
                random_state=42
            )))
        ])
        
    def fit(self, X, y):
        """Train the classifier on a multi-label text dataset."""
        print(f"Training ToxicityClassifier on {len(X)} samples across {len(self.categories)} categories...")
        self.pipeline.fit(X, y)
        print("Training completed successfully!")
        
    def predict_proba(self, X):
        """Predict multi-label class probabilities. Returns a dict of arrays for each class or list of dicts."""
        # MultiOutputClassifier.predict_proba returns a list of arrays (one per class)
        # where each array has shape (n_samples, 2) representing [prob_class_0, prob_class_1]
        probs_list = self.pipeline.predict_proba(X)
        
        n_samples = len(X)
        results = []
        
        for i in range(n_samples):
            sample_probs = {}
            for class_idx, class_name in enumerate(self.categories):
                # probability of class being 1
                sample_probs[class_name] = float(probs_list[class_idx][i][1])
            results.append(sample_probs)
            
        return results
        
    def predict(self, X):
        """Predict labels based on customizable decision thresholds."""
        prob_dicts = self.predict_proba(X)
        predictions = []
        
        for prob_dict in prob_dicts:
            pred_dict = {}
            for col in self.categories:
                # Apply class-specific threshold
                threshold = self.thresholds.get(col, 0.5)
                pred_dict[col] = 1 if prob_dict[col] >= threshold else 0
            predictions.append(pred_dict)
            
        return predictions
        
    def optimize_thresholds(self, X_val, y_val, target_precision=0.80, target_class='severe_toxic'):
        """Tune decision thresholds to achieve a specific target precision on validation data."""
        print(f"Optimizing decision thresholds. Target: >={target_precision * 100}% precision on '{target_class}'...")
        prob_dicts = self.predict_proba(X_val)
        
        # Convert prob_dicts to numpy array of shape (n_samples, n_classes)
        probs = np.zeros((len(X_val), len(self.categories)))
        for i, prob_dict in enumerate(prob_dicts):
            for j, col in enumerate(self.categories):
                probs[i, j] = prob_dict[col]
                
        # Find index of target class
        target_idx = self.categories.index(target_class)
        target_true = y_val[:, target_idx]
        target_probs = probs[:, target_idx]
        
        best_threshold = 0.5
        max_f1 = 0.0
        best_threshold_for_f1 = 0.5
        target_met = False
        achieved_precision = 0.0
        achieved_recall = 0.0
        
        # Try thresholds from 0.05 to 0.99
        for th in np.linspace(0.05, 0.99, 100):
            preds = (target_probs >= th).astype(int)
            tp = np.sum((preds == 1) & (target_true == 1))
            fp = np.sum((preds == 1) & (target_true == 0))
            fn = np.sum((preds == 0) & (target_true == 1))
            
            if (tp + fp) > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Keep track of absolute maximum F1 score
                if f1 > max_f1:
                    max_f1 = f1
                    best_threshold_for_f1 = th
                    
                # If target is met, select the lowest threshold that meets it (to preserve recall)
                if precision >= target_precision and not target_met:
                    best_threshold = th
                    achieved_precision = precision
                    achieved_recall = recall
                    target_met = True
        
        if not target_met:
            # Fall back to the threshold that maximizes F1-score
            best_threshold = best_threshold_for_f1
            preds = (target_probs >= best_threshold).astype(int)
            tp = np.sum((preds == 1) & (target_true == 1))
            fp = np.sum((preds == 1) & (target_true == 0))
            fn = np.sum((preds == 0) & (target_true == 1))
            achieved_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            achieved_recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0
            print(f"⚠️ Target precision of {target_precision:.2%} could not be fully met. Falling back to the threshold maximizing F1-score.")
            
        self.thresholds[target_class] = float(best_threshold)
        print(f"Optimization complete! Best threshold for '{target_class}': {best_threshold:.4f} (Achieved Precision: {achieved_precision:.2%}, Recall: {achieved_recall:.2%})")
        return best_threshold
        
    def save(self, filepath):
        """Save vectorizer, model, and thresholds."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'pipeline': self.pipeline,
            'categories': self.categories,
            'thresholds': self.thresholds
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved successfully to {filepath}")
        
    @classmethod
    def load(cls, filepath):
        """Load a saved model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        model_data = joblib.load(filepath)
        classifier = cls(categories=model_data['categories'])
        classifier.pipeline = model_data['pipeline']
        classifier.thresholds = model_data['thresholds']
        print(f"Model loaded successfully from {filepath}")
        return classifier


class DeepLearningToxicityClassifier:
    """Wrapper class for Hugging Face HuggingFace model unitary/toxic-bert or similar."""
    def __init__(self, model_name='unitary/toxic-bert'):
        self.model_name = model_name
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.pipeline = None
        self.initialized = False
        
    def initialize(self):
        """Load Hugging Face transformers pipeline only when requested (for speed and CPU/GPU memory safety)."""
        if self.initialized:
            return
            
        print(f"Initializing deep learning model: {self.model_name}...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            import torch
            
            device = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Use top_k=None (modern standard), fallback to return_all_scores=True for backwards compatibility
            try:
                self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, device=device)
            except Exception:
                self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
                
            self.initialized = True
            print("Deep Learning model loaded successfully!")
        except Exception as e:
            print(f"Error loading deep learning model: {e}")
            raise e
            
    def predict_proba(self, texts: list[str]):
        """Predict label probabilities using the sequence classification model."""
        if not self.initialized:
            self.initialize()
            
        preds = self.pipeline(texts)
        
        # Robustly handle different Hugging Face return formats (flat lists vs nested lists)
        if isinstance(preds, dict):
            preds = [[preds]]
        elif isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
            preds = [preds]
            
        results = []
        for sample_pred in preds:
            prob_dict = {}
            for label_score in sample_pred:
                # toxic-bert uses labels: 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
                label = label_score['label']
                score = label_score['score']
                prob_dict[label] = score
            results.append(prob_dict)
        return results
        
    def predict(self, texts: list[str], thresholds=None):
        """Predict labels based on thresholds."""
        prob_dicts = self.predict_proba(texts)
        if thresholds is None:
            thresholds = {
                'toxic': 0.5,
                'severe_toxic': 0.5,
                'obscene': 0.5,
                'threat': 0.5,
                'insult': 0.5,
                'identity_hate': 0.5
            }
            
        predictions = []
        for prob_dict in prob_dicts:
            pred_dict = {}
            for col in self.categories:
                threshold = thresholds.get(col, 0.5)
                pred_dict[col] = 1 if prob_dict.get(col, 0.0) >= threshold else 0
            predictions.append(pred_dict)
        return predictions
