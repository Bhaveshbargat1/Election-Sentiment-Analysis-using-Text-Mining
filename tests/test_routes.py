import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_dashboard_route(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Election Sentiment Mining Dashboard" in res.data
    assert b"Candidate Sentiment Scoreboard" in res.data

def test_predict_get(client):
    res = client.get("/predict")
    assert res.status_code == 200
    assert b"Live Sentiment Predictor" in res.data

def test_predict_post(client):
    res = client.post("/predict", data={"tweet_text": "Inspiring speech by Candidate A today!"})
    assert res.status_code == 200
    assert b"Sentiment Analysis Verdict" in res.data
    assert b"Text Mining &amp; NLP Preprocessing Pipeline" in res.data or b"Text Mining & NLP Preprocessing Pipeline" in res.data

def test_candidates_route(client):
    res = client.get("/candidates")
    assert res.status_code == 200
    assert b"Candidate &amp; Political Party Sentiment Analytics" in res.data or b"Candidate & Political Party" in res.data

def test_explorer_route(client):
    res = client.get("/explorer")
    assert res.status_code == 200
    assert b"Social Media Dataset Explorer" in res.data

def test_evaluation_route(client):
    res = client.get("/evaluation")
    assert res.status_code == 200
    assert b"Machine Learning Model Evaluation &amp; Benchmarks" in res.data or b"Model Evaluation" in res.data

def test_analytics_route(client):
    res = client.get("/analytics")
    assert res.status_code == 200
    assert b"Text Analytics &amp; Architecture" in res.data or b"Text Analytics" in res.data

def test_api_predict(client):
    payload = {"text": "Huge victory and outstanding rally today for Candidate A!"}
    res = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "ml_prediction" in data["data"]
    assert data["data"]["ml_prediction"]["sentiment"] == "Positive"

def test_api_stats(client):
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["total_tweets"] > 0
