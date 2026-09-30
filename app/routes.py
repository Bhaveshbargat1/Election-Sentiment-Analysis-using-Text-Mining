"""
Application Routes and REST API Blueprint
Handles views for Dashboard, Live Predictor, Candidate Analytics,
Dataset Explorer, Model Benchmarking, and JSON API endpoints.
"""

import os
import json
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, current_app
from app.sentiment_analyzer import SentimentAnalyzer

bp = Blueprint("main", __name__)
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer(current_app.config["MODELS_DIR"])
    return _analyzer

def get_dataset():
    data_path = current_app.config["DATA_PATH"]
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()

def get_metrics():
    metrics_path = os.path.join(current_app.config["MODELS_DIR"], "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@bp.route("/")
def index():
    """Dashboard Overview showing key high-level statistics."""
    df = get_dataset()
    metrics_data = get_metrics()
    
    total_tweets = len(df)
    sentiment_counts = df["ground_truth_sentiment"].value_counts().to_dict() if not df.empty else {}
    
    pos_count = sentiment_counts.get("Positive", 0)
    neg_count = sentiment_counts.get("Negative", 0)
    neu_count = sentiment_counts.get("Neutral", 0)
    
    pos_pct = round((pos_count / total_tweets * 100), 1) if total_tweets else 0
    neg_pct = round((neg_count / total_tweets * 100), 1) if total_tweets else 0
    neu_pct = round((neu_count / total_tweets * 100), 1) if total_tweets else 0
    
    # Candidate summaries
    candidate_stats = []
    if not df.empty and "candidate_mentioned" in df.columns:
        for cand in sorted(df["candidate_mentioned"].unique()):
            sub = df[df["candidate_mentioned"] == cand]
            c_total = len(sub)
            c_pos = len(sub[sub["ground_truth_sentiment"] == "Positive"])
            c_neg = len(sub[sub["ground_truth_sentiment"] == "Negative"])
            c_neu = len(sub[sub["ground_truth_sentiment"] == "Neutral"])
            nss = round(((c_pos - c_neg) / c_total * 100), 1) if c_total else 0
            candidate_stats.append({
                "candidate": cand,
                "party": sub["political_party"].iloc[0] if "political_party" in sub.columns else "N/A",
                "total": c_total,
                "positive": c_pos,
                "negative": c_neg,
                "neutral": c_neu,
                "net_sentiment_score": nss
            })
            
    # Top topics
    topic_stats = []
    if not df.empty and "election_topic" in df.columns:
        topic_counts = df["election_topic"].value_counts().head(5).to_dict()
        for t, cnt in topic_counts.items():
            topic_stats.append({"topic": t, "count": cnt})
            
    best_model_name = metrics_data.get("best_model_name", "Logistic Regression")
    best_model_acc = 100.0
    if best_model_name in metrics_data.get("models", {}):
        best_model_acc = round(metrics_data["models"][best_model_name]["accuracy"] * 100, 2)
        
    return render_template(
        "index.html",
        total_tweets=total_tweets,
        pos_count=pos_count,
        neg_count=neg_count,
        neu_count=neu_count,
        pos_pct=pos_pct,
        neg_pct=neg_pct,
        neu_pct=neu_pct,
        candidate_stats=candidate_stats,
        topic_stats=topic_stats,
        best_model_name=best_model_name,
        best_model_acc=best_model_acc
    )

@bp.route("/predict", methods=["GET", "POST"])
def predict():
    """Interactive Live Sentiment Mining and Pipeline Inspector."""
    result = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("tweet_text", "")
        if input_text.strip():
            analyzer = get_analyzer()
            result = analyzer.analyze(input_text)
            
    return render_template("predict.html", result=result, input_text=input_text)

@bp.route("/candidates")
def candidates():
    """Comparative candidate and political party sentiment analytics."""
    df = get_dataset()
    candidates_list = []
    
    if not df.empty and "candidate_mentioned" in df.columns:
        for cand in sorted(df["candidate_mentioned"].unique()):
            sub = df[df["candidate_mentioned"] == cand]
            total = len(sub)
            pos = len(sub[sub["ground_truth_sentiment"] == "Positive"])
            neg = len(sub[sub["ground_truth_sentiment"] == "Negative"])
            neu = len(sub[sub["ground_truth_sentiment"] == "Neutral"])
            
            party = sub["political_party"].iloc[0] if "political_party" in sub.columns else "N/A"
            avg_likes = int(sub["likes_count"].mean()) if "likes_count" in sub.columns else 0
            avg_retweets = int(sub["retweets_count"].mean()) if "retweets_count" in sub.columns else 0
            nss = round(((pos - neg) / total * 100), 1) if total else 0
            
            candidates_list.append({
                "name": cand,
                "party": party,
                "total_mentions": total,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "pos_pct": round(pos/total*100, 1) if total else 0,
                "neg_pct": round(neg/total*100, 1) if total else 0,
                "neu_pct": round(neu/total*100, 1) if total else 0,
                "net_sentiment_score": nss,
                "avg_likes": avg_likes,
                "avg_retweets": avg_retweets
            })
            
    return render_template("candidates.html", candidates=candidates_list)

@bp.route("/explorer")
def explorer():
    """Dataset explorer with live search and filtering."""
    df = get_dataset()
    
    # Query parameters
    candidate_filter = request.args.get("candidate", "All")
    sentiment_filter = request.args.get("sentiment", "All")
    search_query = request.args.get("q", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 20
    
    filtered_df = df.copy() if not df.empty else pd.DataFrame()
    
    if not filtered_df.empty:
        if candidate_filter != "All":
            filtered_df = filtered_df[filtered_df["candidate_mentioned"] == candidate_filter]
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df["ground_truth_sentiment"] == sentiment_filter]
        if search_query:
            filtered_df = filtered_df[filtered_df["raw_tweet_text"].str.contains(search_query, case=False, na=False)]
            
    total_records = len(filtered_df)
    total_pages = max(1, (total_records + per_page - 1) // per_page)
    page = min(page, total_pages)
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    page_records = filtered_df.iloc[start_idx:end_idx].to_dict(orient="records") if not filtered_df.empty else []
    
    all_candidates = ["All"] + sorted(df["candidate_mentioned"].unique().tolist()) if not df.empty else ["All"]
    all_sentiments = ["All", "Positive", "Negative", "Neutral"]
    
    return render_template(
        "explorer.html",
        records=page_records,
        total_records=total_records,
        page=page,
        total_pages=total_pages,
        candidate_filter=candidate_filter,
        sentiment_filter=sentiment_filter,
        search_query=search_query,
        all_candidates=all_candidates,
        all_sentiments=all_sentiments
    )

@bp.route("/evaluation")
def evaluation():
    """Model evaluation benchmarks, classification metrics, and confusion matrix."""
    metrics_data = get_metrics()
    models = metrics_data.get("models", {})
    best_model_name = metrics_data.get("best_model_name", "N/A")
    classes = metrics_data.get("classes", ["Negative", "Neutral", "Positive"])
    
    return render_template(
        "evaluation.html",
        models=models,
        best_model_name=best_model_name,
        classes=classes
    )

@bp.route("/analytics")
def analytics():
    """Text mining analytics, n-grams, word clouds, and architecture diagram."""
    return render_template("analytics.html")

# ----------------- REST API ENDPOINTS -----------------

@bp.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for real-time sentiment prediction.
    Accepts JSON: { "text": "tweet content" }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"status": "error", "message": "Missing 'text' field in JSON request"}), 400
        
    text = data["text"]
    analyzer = get_analyzer()
    analysis = analyzer.analyze(text)
    return jsonify({"status": "success", "data": analysis})

@bp.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns dataset summary statistics in JSON."""
    df = get_dataset()
    if df.empty:
        return jsonify({"status": "error", "message": "No dataset found"}), 404
        
    counts = df["ground_truth_sentiment"].value_counts().to_dict()
    cand_counts = df["candidate_mentioned"].value_counts().to_dict()
    
    return jsonify({
        "status": "success",
        "total_tweets": len(df),
        "sentiment_counts": counts,
        "candidate_counts": cand_counts
    })
