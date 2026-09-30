"""
Model Trainer Module
Trains and compares multiple classification models for election text mining:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (LinearSVC with calibration for probabilities)
- Random Forest Classifier
Computes comprehensive evaluation metrics, confusion matrices, and exports production models.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from app.text_preprocessor import TextPreprocessor

class ModelTrainer:
    def __init__(self, raw_data_path: str, models_dir: str, static_img_dir: str):
        self.raw_data_path = raw_data_path
        self.models_dir = models_dir
        self.static_img_dir = static_img_dir
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3500,
            sublinear_tf=True
        )
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=1.5, random_state=42),
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
            "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42)),
            "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=20, random_state=42)
        }
        self.metrics = {}
        self.best_model_name = None
        self.best_model = None

    def preprocess_dataset(self) -> pd.DataFrame:
        """Loads raw dataset, cleans text, and saves processed dataset."""
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(df)} tweets from {self.raw_data_path}")
        
        # Apply preprocessing pipeline
        df["processed_text"] = df["raw_tweet_text"].apply(self.preprocessor.transform)
        
        # Save processed dataset
        processed_path = os.path.join(os.path.dirname(self.raw_data_path), "election_tweets_processed.csv")
        df.to_csv(processed_path, index=False, encoding="utf-8")
        print(f"Processed dataset saved to {processed_path}")
        return df

    def train_and_evaluate(self):
        """Preprocesses data, trains all models, evaluates, and selects best model."""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.static_img_dir, exist_ok=True)
        
        df = self.preprocess_dataset()
        
        X = df["processed_text"]
        y = df["ground_truth_sentiment"]
        
        # 80-20 Train-Test Split stratified by sentiment
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")
        
        # TF-IDF Feature Extraction
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        best_f1 = -1.0
        classes = sorted(y.unique())
        
        for name, clf in self.models.items():
            print(f"\n--- Training {name} ---")
            clf.fit(X_train_tfidf, y_train)
            y_pred = clf.predict(X_test_tfidf)
            
            acc = accuracy_score(y_test, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            
            self.metrics[name] = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "macro_precision": round(float(macro_prec), 4),
                "macro_recall": round(float(macro_rec), 4),
                "macro_f1": round(float(macro_f1), 4),
                "confusion_matrix": cm.tolist(),
                "classes": classes,
                "detailed_report": report
            }
            
            print(f"{name} -> Accuracy: {acc*100:.2f}%, F1-Score: {f1*100:.2f}%")
            
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_model = clf
                
        print(f"\nBest Model: {self.best_model_name} with Weighted F1-Score of {best_f1*100:.2f}%")
        
        # Save best model, vectorizer, and metrics
        model_save_path = os.path.join(self.models_dir, "best_sentiment_model.joblib")
        vec_save_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        metrics_save_path = os.path.join(self.models_dir, "model_metrics.json")
        
        joblib.dump(self.best_model, model_save_path)
        joblib.dump(self.vectorizer, vec_save_path)
        
        export_metrics = {
            "best_model_name": self.best_model_name,
            "classes": classes,
            "models": self.metrics
        }
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            json.dump(export_metrics, f, indent=4)
            
        print(f"Artifacts saved in {self.models_dir}")
        
        # Generate Visualizations
        self.plot_model_comparison()
        self.plot_confusion_matrix(classes)
        
        return export_metrics

    def plot_model_comparison(self):
        """Creates a bar chart comparing accuracy and F1 scores of all models."""
        plt.figure(figsize=(9, 5), dpi=300)
        
        model_names = list(self.metrics.keys())
        accuracies = [self.metrics[m]["accuracy"] * 100 for m in model_names]
        f1_scores = [self.metrics[m]["f1_score"] * 100 for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2563eb', alpha=0.9)
        rects2 = ax.bar(x + width/2, f1_scores, width, label='Weighted F1-Score (%)', color='#10b981', alpha=0.9)
        
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('Machine Learning Models Performance Comparison - Election Sentiment Mining', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=11, fontweight='medium')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Attach values above bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        out_path = os.path.join(self.static_img_dir, "model_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Model comparison chart saved at: {out_path}")

    def plot_confusion_matrix(self, classes):
        """Plots confusion matrix heatmap for the best performing model."""
        best_cm = np.array(self.metrics[self.best_model_name]["confusion_matrix"])
        
        plt.figure(figsize=(7, 5.5), dpi=300)
        sns.heatmap(
            best_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            annot_kws={"size": 12, "weight": "bold"}
        )
        plt.title(f'Confusion Matrix: {self.best_model_name}', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Predicted Sentiment Class', fontsize=11, fontweight='bold')
        plt.ylabel('Actual (Ground Truth) Class', fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        out_path = os.path.join(self.static_img_dir, "confusion_matrix.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Confusion matrix plot saved at: {out_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_csv = os.path.join(base_dir, "data", "election_tweets_raw.csv")
    models_path = os.path.join(base_dir, "models")
    static_img = os.path.join(base_dir, "app", "static", "images")
    
    trainer = ModelTrainer(raw_csv, models_path, static_img)
    trainer.train_and_evaluate()
