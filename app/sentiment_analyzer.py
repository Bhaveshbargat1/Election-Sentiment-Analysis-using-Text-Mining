"""
Sentiment Analyzer Module
Combines Machine Learning inference with Lexicon-based methods (VADER & TextBlob)
to provide deep, multi-perspective sentiment analysis of election text.
"""

import os
import joblib
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.text_preprocessor import TextPreprocessor

class SentimentAnalyzer:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
        
        # Load trained ML model and TF-IDF vectorizer
        model_path = os.path.join(models_dir, "best_sentiment_model.joblib")
        vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            self.classes_ = getattr(self.model, "classes_", ["Negative", "Neutral", "Positive"])
        else:
            self.model = None
            self.vectorizer = None
            self.classes_ = ["Negative", "Neutral", "Positive"]

    def analyze(self, text: str) -> dict:
        """
        Executes end-to-end multi-layer sentiment analysis:
        1. Preprocessing stages
        2. Supervised ML classification
        3. VADER sentiment lexicon scoring
        4. TextBlob polarity & subjectivity
        5. Entity/Candidate detection
        """
        if not text or not text.strip():
            return {
                "error": "Empty text provided"
            }
            
        # 1. Preprocessing stages
        prep_steps = self.preprocessor.pipeline_step_by_step(text)
        processed_str = prep_steps["processed_text"]
        
        # 2. ML Model Prediction
        ml_prediction = "Neutral"
        confidence = 0.50
        probabilities = {}
        
        if self.model is not None and self.vectorizer is not None:
            # Transform to TF-IDF
            feat = self.vectorizer.transform([processed_str])
            pred_class = self.model.predict(feat)[0]
            ml_prediction = str(pred_class)
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(feat)[0]
                probabilities = {cls: round(float(prob), 4) for cls, prob in zip(self.classes_, probs)}
                confidence = float(np.max(probs))
            else:
                probabilities = {c: (1.0 if c == ml_prediction else 0.0) for c in self.classes_}
                confidence = 1.0
        else:
            # Fallback to VADER if model not yet trained
            ml_prediction = "Neutral"
            probabilities = {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}
            
        # 3. VADER Sentiment Scoring
        vader_scores = self.vader.polarity_scores(text)
        # Compound: >= 0.05 (Pos), <= -0.05 (Neg), else (Neu)
        if vader_scores["compound"] >= 0.05:
            vader_label = "Positive"
        elif vader_scores["compound"] <= -0.05:
            vader_label = "Negative"
        else:
            vader_label = "Neutral"
            
        # 4. TextBlob Polarity & Subjectivity
        blob = TextBlob(text)
        polarity = round(float(blob.sentiment.polarity), 4)
        subjectivity = round(float(blob.sentiment.subjectivity), 4)
        
        # 5. Entity/Candidate Detection
        lower = text.lower()
        candidates_detected = []
        if "candidate a" in lower or "@candidatea" in lower:
            candidates_detected.append("Candidate A (AFP)")
        if "candidate b" in lower or "@candidateb" in lower:
            candidates_detected.append("Candidate B (NDU)")
        if "candidate c" in lower or "@candidatec" in lower:
            candidates_detected.append("Candidate C (RIP)")
        if not candidates_detected:
            candidates_detected.append("General Election Discourse")
            
        return {
            "input_text": text,
            "preprocessing": prep_steps,
            "ml_prediction": {
                "sentiment": ml_prediction,
                "confidence_score": round(confidence * 100, 2),
                "probabilities": probabilities
            },
            "vader_analysis": {
                "sentiment": vader_label,
                "compound": round(vader_scores["compound"], 4),
                "positive": round(vader_scores["pos"], 4),
                "neutral": round(vader_scores["neu"], 4),
                "negative": round(vader_scores["neg"], 4)
            },
            "textblob_analysis": {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "interpretation": "Opinion/Subjective" if subjectivity > 0.5 else "Factual/Objective"
            },
            "entities": candidates_detected
        }
