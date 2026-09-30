# Election Social Media Sentiment Analysis using Text Mining

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.10-336699?style=for-the-badge)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end Data Mining and Natural Language Processing (NLP) system designed to capture, clean, preprocess, analyze, and visualize public voter sentiment during an election cycle. The platform combines supervised machine learning classifiers (Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest) with lexicon-based approaches (VADER, TextBlob) wrapped in an interactive Flask web dashboard and REST API.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Folder Structure](#-folder-structure)
- [Text Mining & NLP Pipeline](#-text-mining--nlp-pipeline)
- [Machine Learning Models & Evaluation](#-machine-learning-models--evaluation)
- [Installation Guide](#-installation-guide)
- [Running the Project](#-running-the-project)
- [REST API Endpoints](#-rest-api-endpoints)
- [Automated Testing](#-automated-testing)
- [Academic Information](#-academic-information)

---

## 🚀 Project Overview

During national and local election campaigns, millions of citizens share opinions, debates, and concerns across social media platforms like X (Twitter). Understanding voter behavior and real-time opinion shifts is paramount for electoral analysts, media organizations, and policy researchers.

This project delivers:
1. **Multi-Stage Text Preprocessor**: Cleans raw social media noise (URLs, mentions, emojis, slang, contractions, and punctuation) while preserving critical domain context (such as hashtags and sentiment keywords).
2. **Feature Engineering**: Employs TF-IDF (Term Frequency-Inverse Document Frequency) unigram and bigram token weighting alongside sentiment lexicon scoring.
3. **Multi-Model Sentiment Classification**: Compares four supervised classification algorithms alongside VADER and TextBlob to provide robust sentiment decisions with confidence scores.
4. **Interactive Web Dashboard**: Built with Flask and Bootstrap 5, featuring a live text miner, candidate scoreboard, dataset explorer with multi-factor filtering, model performance benchmarks, and word clouds.
5. **REST API**: Production-ready programmatic endpoints for downstream application integration.

---

## ✨ Key Features

- **Real-Time Sentiment Predictor & Inspector**: Input any custom statement or tweet to view both the sentiment verdict and the step-by-step transformations across every stage of the NLP pipeline.
- **Candidate Scoreboard & Net Sentiment Score (NSS)**: Automatically estimates net approval percentage:
  $$\text{NSS} = \% \text{Positive Mentions} - \% \text{Negative Mentions}$$
- **Multi-Perspective Ensemble Scoring**:
  - *Supervised ML Probability*: Machine-learned classification probability based on TF-IDF representation.
  - *VADER Lexicon*: Valence-aware rule-based scoring accounting for capitalization and punctuation intensity.
  - *TextBlob*: Polarity (-1.0 to +1.0) and Subjectivity (Objective vs. Opinion) analysis.
- **Dataset Explorer**: Paginated search and filtering across candidates, sentiments, and topics.
- **Visual Analytics**: High-resolution Word Clouds for positive and negative discourse, class distribution charts, candidate comparative graphs, and confusion matrix heatmaps.

---

## 🏗️ System Architecture

The following flowchart outlines the entire text mining workflow:

```
[Social Media Stream] 
       │ (Raw Election Tweets, Mentions & Hashtags)
       ▼
[Text Preprocessing Pipeline] 
       │ (Contraction Expansion, Regex Cleaning, Tokenization, Lemmatization)
       ▼
[Feature Extraction] 
       │ (TF-IDF N-grams & VADER Lexicon Scoring)
       ▼
[Machine Learning Sentiment Classifiers] 
       │ (Logistic Regression, Naive Bayes, Linear SVM, Random Forest)
       ▼
[Analytics & Aggregation Layer] 
       │ (Candidate Approval Trends, NSS, Frequency Clouds)
       ▼
[Flask Web Application & REST API] 
       │ (Dashboard, Live Predictor, Explorer, Benchmarks)
       ▼
[End User Interface (Bootstrap 5 Web Browser)]
```

*(Full high-resolution architecture diagram is viewable in `app/static/images/architecture_diagram.png`)*.

---

## 📁 Folder Structure

```text
Election-Sentiment-Analysis-using-Text-Mining/
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
├── run.py                          # Flask application launcher script
├── app/
│   ├── __init__.py                 # Flask app factory
│   ├── routes.py                   # Page controllers and REST API endpoints
│   ├── text_preprocessor.py        # Regex cleaning, tokenizer, stopword filter, lemmatizer
│   ├── sentiment_analyzer.py       # Hybrid ML + VADER + TextBlob inference engine
│   ├── model_trainer.py            # Supervised training, metrics, and chart generation
│   ├── visualizer.py               # Generates wordclouds, charts, and architecture diagrams
│   ├── templates/                  # Jinja2 HTML templates
│   │   ├── base.html               # Navigation and footer layout
│   │   ├── index.html              # Main statistics dashboard
│   │   ├── predict.html            # Interactive live predictor & pipeline inspector
│   │   ├── candidates.html         # Candidate comparison & engagement stats
│   │   ├── explorer.html           # Searchable dataset explorer
│   │   ├── evaluation.html         # Model benchmarks & confusion matrix
│   │   └── analytics.html          # Word clouds & architecture flowchart
│   └── static/
│       ├── css/
│       │   └── style.css           # Custom modern UI styling
│       ├── js/
│       │   └── app.js              # Client-side JavaScript interactions
│       └── images/                 # Generated plots and diagrams
│           ├── architecture_diagram.png
│           ├── candidate_sentiment.png
│           ├── confusion_matrix.png
│           ├── model_comparison.png
│           ├── sentiment_distribution.png
│           ├── wordcloud_negative.png
│           └── wordcloud_positive.png
├── data/
│   ├── generate_dataset.py         # Reproducible dataset synthesizer script
│   ├── election_tweets_raw.csv     # Raw dataset (1500 records)
│   └── election_tweets_processed.csv # Tokenized and cleaned dataset
├── models/
│   ├── best_sentiment_model.joblib # Serialized production classifier
│   ├── tfidf_vectorizer.joblib     # Serialized TF-IDF vectorizer
│   └── model_metrics.json          # Benchmark statistics and confusion matrix data
└── tests/
    ├── __init__.py
    ├── test_preprocessor.py        # Preprocessing unit tests
    ├── test_sentiment.py           # Sentiment engine unit tests
    └── test_routes.py              # Routes and REST API integration tests
```

---

## 🔬 Text Mining & NLP Pipeline

Every raw social media post passes through a rigorous six-stage pipeline:

1. **Contraction Expansion**: Replaces colloquial contractions (`can't` &rarr; `cannot`, `won't` &rarr; `will not`).
2. **Regex Cleaning**: Strips web links (`https?://\S+`), user mentions (`@user`), digits, and special characters.
3. **Hashtag Decomposition**: Extracts the semantic keyword from hashtags (`#Election2026` &rarr; `election`).
4. **Tokenization**: Segments cleaned sentences into lowercase atomic word tokens.
5. **Stop Word Removal**: Eliminates non-informative words using the standard NLTK English stop words corpus.
6. **Lemmatization**: Employs WordNet Lemmatizer to reduce inflected variants to their root base lemma (`promises` &rarr; `promise`, `debating` &rarr; `debate`).

---

## 📊 Machine Learning Models & Evaluation

The system trains and compares four supervised classification algorithms using an 80/20 stratified split:

| Algorithm / Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **Champion (Deployed)** |
| **Linear SVM (Calibrated)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | Evaluated |
| **Multinomial Naive Bayes** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | Evaluated |
| **Random Forest** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | Evaluated |

*(Benchmark charts and confusion matrices are generated dynamically into `app/static/images/`)*.

---

## ⚙️ Installation Guide

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Git

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bhaveshbargat1/Election-Sentiment-Analysis-using-Text-Mining.git
   cd Election-Sentiment-Analysis-using-Text-Mining
   ```

2. **Create and Activate Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Corpora**:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('punkt_tab')"
   ```

---

## 🏃 Running the Project

Start the local Flask development web server:

```bash
python run.py
```

Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

To run with custom host or port:
```bash
python run.py --host 0.0.0.0 --port 8080
```

---

## 🌐 REST API Endpoints

### 1. Predict Sentiment
- **URL**: `/api/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "text": "Candidate A gave an inspiring town hall speech today! Lower prescription costs and 500k new green jobs is visionary."
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "data": {
      "input_text": "Candidate A gave an inspiring town hall speech...",
      "ml_prediction": {
        "sentiment": "Positive",
        "confidence_score": 90.19,
        "probabilities": {
          "Negative": 0.0565,
          "Neutral": 0.0416,
          "Positive": 0.9019
        }
      },
      "vader_analysis": {
        "sentiment": "Positive",
        "compound": 0.8999
      },
      "textblob_analysis": {
        "polarity": 0.375,
        "subjectivity": 0.6667
      },
      "entities": ["Candidate A (AFP)"]
    }
  }
  ```

### 2. Dataset Aggregate Stats
- **URL**: `/api/stats`
- **Method**: `GET`
- **Response**: Returns total volume, sentiment counts, and candidate breakdown.

---

## 🧪 Automated Testing

Execute the automated test suite with pytest:

```bash
pytest -v
```

All 18 unit and integration tests will run:
- Preprocessor contraction expansion, cleaning, tokenization, lemmatization
- Multi-model inference, edge cases (empty strings), and entity extraction
- Flask HTTP routes, template rendering, and REST API endpoints

---

## 🎓 Academic Information

- **Course / Subject**: Data Mining (Tutorial No. 1)
- **Institution**: Priyadarshini College of Engineering, Nagpur (Autonomous Institute Affiliated to R.T.M. Nagpur University)
- **Department**: Department of Artificial Intelligence & Data Science
- **Student Name**: Bhavesh Bargat
- **Roll Number**: 235
- **Degree**: B.Tech VIIth Semester
- **Supervisor / Guide**: Ms. Heena Kachhela
- **Academic Year**: 2026-27

---

## 📄 License
This project is open-source and distributed under the [MIT License](LICENSE).
