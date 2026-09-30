"""
Visualizer Module
Generates high-resolution publication-quality visualizations for the web application
and documentation:
1. Positive Sentiment Word Cloud
2. Negative Sentiment Word Cloud
3. Overall Sentiment Distribution
4. Candidate Comparative Sentiment Chart
5. End-to-End System Architecture Diagram Flowchart
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from wordcloud import WordCloud

class Visualizer:
    def __init__(self, data_path: str, output_img_dir: str):
        self.data_path = data_path
        self.output_img_dir = output_img_dir
        os.makedirs(output_img_dir, exist_ok=True)
        
    def generate_all(self):
        """Generates all analytics plots and diagrams."""
        df = pd.read_csv(self.data_path)
        print("Generating analytics plots...")
        self.generate_wordclouds(df)
        self.generate_sentiment_distribution(df)
        self.generate_candidate_sentiment_chart(df)
        self.generate_architecture_diagram()
        print("All visual assets successfully generated.")

    def generate_wordclouds(self, df: pd.DataFrame):
        """Generates separate word clouds for positive and negative election tweets."""
        pos_text = " ".join(df[df["ground_truth_sentiment"] == "Positive"]["processed_text"].dropna())
        neg_text = " ".join(df[df["ground_truth_sentiment"] == "Negative"]["processed_text"].dropna())
        
        # Positive Word Cloud
        wc_pos = WordCloud(
            width=1000, height=550,
            background_color='white',
            colormap='Greens',
            max_words=120,
            collocations=False
        ).generate(pos_text)
        
        plt.figure(figsize=(10, 5.5), dpi=300)
        plt.imshow(wc_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('Election Social Media: Positive Sentiment Key Terms', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        out_pos = os.path.join(self.output_img_dir, "wordcloud_positive.png")
        plt.savefig(out_pos, dpi=300)
        plt.close()
        print(f"Positive WordCloud saved: {out_pos}")
        
        # Negative Word Cloud
        wc_neg = WordCloud(
            width=1000, height=550,
            background_color='white',
            colormap='Reds',
            max_words=120,
            collocations=False
        ).generate(neg_text)
        
        plt.figure(figsize=(10, 5.5), dpi=300)
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Election Social Media: Negative Sentiment Key Terms', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        out_neg = os.path.join(self.output_img_dir, "wordcloud_negative.png")
        plt.savefig(out_neg, dpi=300)
        plt.close()
        print(f"Negative WordCloud saved: {out_neg}")

    def generate_sentiment_distribution(self, df: pd.DataFrame):
        """Generates distribution pie & bar chart for sentiment classes."""
        counts = df["ground_truth_sentiment"].value_counts()
        labels = counts.index.tolist()
        sizes = counts.values.tolist()
        colors = ['#10b981', '#ef4444', '#6b7280']  # green, red, gray
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
        
        # Donut Chart
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=140, colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
            textprops=dict(fontsize=11, fontweight='bold')
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(11)
            at.set_weight('bold')
        ax1.set_title('Overall Sentiment Share (%)', fontsize=13, fontweight='bold', pad=10)
        
        # Bar Chart
        bars = ax2.bar(labels, sizes, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
        ax2.set_ylabel('Number of Tweets', fontsize=11, fontweight='bold')
        ax2.set_title('Tweet Count by Sentiment Class', fontsize=13, fontweight='bold', pad=10)
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{int(yval)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        plt.suptitle('Election Sentiment Mining Dataset Distribution Overview', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_img_dir, "sentiment_distribution.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sentiment distribution plot saved: {out_path}")

    def generate_candidate_sentiment_chart(self, df: pd.DataFrame):
        """Generates comparative breakdown of sentiment across candidates."""
        ct = pd.crosstab(df["candidate_mentioned"], df["ground_truth_sentiment"])
        
        candidates = ct.index.tolist()
        pos = ct["Positive"].tolist() if "Positive" in ct else [0]*len(candidates)
        neu = ct["Neutral"].tolist() if "Neutral" in ct else [0]*len(candidates)
        neg = ct["Negative"].tolist() if "Negative" in ct else [0]*len(candidates)
        
        x = np.arange(len(candidates))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        
        r1 = ax.bar(x - width, pos, width, label='Positive', color='#10b981', edgecolor='black', linewidth=0.7)
        r2 = ax.bar(x, neu, width, label='Neutral', color='#6b7280', edgecolor='black', linewidth=0.7)
        r3 = ax.bar(x + width, neg, width, label='Negative', color='#ef4444', edgecolor='black', linewidth=0.7)
        
        ax.set_ylabel('Number of Tweets', fontsize=11, fontweight='bold')
        ax.set_title('Comparative Social Media Sentiment by Candidate', fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(candidates, fontsize=11, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.95)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for rects in [r1, r2, r3]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
                            
        plt.tight_layout()
        out_path = os.path.join(self.output_img_dir, "candidate_sentiment.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Candidate sentiment chart saved: {out_path}")

    def generate_architecture_diagram(self):
        """
        Draws a clean professional architecture diagram matching the visual style of the sample PDF.
        """
        fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        
        # Title
        ax.text(50, 95, "System Architecture: Election Social Media Sentiment Analysis using Text Mining",
                ha='center', va='center', fontsize=13, fontweight='bold', color='#1e3a8a')
        
        # Box styling
        box_style = dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1.5)
        box_style_proc = dict(boxstyle="round,pad=0.5", facecolor="#f0fdf4", edgecolor="#10b981", linewidth=1.5)
        box_style_model = dict(boxstyle="round,pad=0.5", facecolor="#fef3c7", edgecolor="#f59e0b", linewidth=1.5)
        box_style_ui = dict(boxstyle="round,pad=0.5", facecolor="#f5f3ff", edgecolor="#8b5cf6", linewidth=1.5)
        
        # Row 1 (y=75): Data Ingestion -> Preprocessing -> Feature Extraction
        # Box 1: Ingestion
        ax.text(18, 75, "Social Media Stream\n(Raw Election Tweets,\nHashtags & Mentions)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#1e293b',
                bbox=box_style, multialignment='center')
        
        # Box 2: Preprocessing
        ax.text(50, 75, "Text Preprocessing Pipeline\n(Cleaning, Contraction Expansion,\nTokenization, Lemmatization)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46',
                bbox=box_style_proc, multialignment='center')
        
        # Box 3: Feature Engineering
        ax.text(82, 75, "Feature Extraction\n(TF-IDF N-Grams &\nVADER Lexicon Scoring)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#92400e',
                bbox=box_style_model, multialignment='center')
                
        # Row 2 (y=45): Classification Models <- Evaluation & Selection <- Sentiment Engine
        # Box 4: ML Models
        ax.text(82, 45, "ML Sentiment Classifiers\n(Logistic Regression, Naive Bayes,\nLinear SVM, Random Forest)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#92400e',
                bbox=box_style_model, multialignment='center')
                
        # Box 5: Analytics Layer
        ax.text(50, 45, "Analytics & Aggregation\n(Candidate Sentiment Trends,\nWord Frequency, Net Score)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#065f46',
                bbox=box_style_proc, multialignment='center')
                
        # Box 6: Web Application
        ax.text(18, 45, "Flask Web Application\n& REST API Engine\n(Inference & Data Service)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#1e293b',
                bbox=box_style, multialignment='center')
                
        # Row 3 (y=16): Interactive UI
        ax.text(50, 16, "Interactive User Interface Dashboard\n(Real-Time Sentiment Predictor | Candidate Trends | Dataset Explorer | Model Metrics)",
                ha='center', va='center', fontsize=10, fontweight='bold', color='#5b21b6',
                bbox=box_style_ui, multialignment='center')
                
        # Arrows with clean styling
        arrow_prop = dict(arrowstyle="->", color="#3b82f6", lw=2, mutation_scale=15)
        arrow_prop_green = dict(arrowstyle="->", color="#10b981", lw=2, mutation_scale=15)
        arrow_prop_amber = dict(arrowstyle="->", color="#f59e0b", lw=2, mutation_scale=15)
        arrow_prop_purple = dict(arrowstyle="->", color="#8b5cf6", lw=2, mutation_scale=15)
        
        # 1 -> 2
        ax.annotate("", xy=(36, 75), xytext=(30, 75), arrowprops=arrow_prop)
        # 2 -> 3
        ax.annotate("", xy=(68, 75), xytext=(64, 75), arrowprops=arrow_prop_green)
        # 3 -> 4
        ax.annotate("", xy=(82, 56), xytext=(82, 64), arrowprops=arrow_prop_amber)
        # 4 -> 5
        ax.annotate("", xy=(66, 45), xytext=(70, 45), arrowprops=arrow_prop_amber)
        # 5 -> 6
        ax.annotate("", xy=(32, 45), xytext=(36, 45), arrowprops=arrow_prop_green)
        # 6 -> 7
        ax.annotate("", xy=(38, 25), xytext=(22, 35), arrowprops=arrow_prop_purple)
        # 5 -> 7
        ax.annotate("", xy=(50, 25), xytext=(50, 35), arrowprops=arrow_prop_purple)
        
        plt.tight_layout()
        out_path = os.path.join(self.output_img_dir, "architecture_diagram.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Architecture diagram saved: {out_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_csv = os.path.join(base_dir, "data", "election_tweets_processed.csv")
    static_images = os.path.join(base_dir, "app", "static", "images")
    vis = Visualizer(data_csv, static_images)
    vis.generate_all()
