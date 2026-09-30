import pytest
from app.sentiment_analyzer import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_positive_sentiment(analyzer):
    text = "Candidate A delivered an outstanding speech on healthcare and job reforms today! Truly inspiring vision."
    res = analyzer.analyze(text)
    assert res["ml_prediction"]["sentiment"] == "Positive"
    assert res["vader_analysis"]["sentiment"] == "Positive"
    assert res["textblob_analysis"]["polarity"] > 0
    assert "Candidate A (AFP)" in res["entities"]

def test_negative_sentiment(analyzer):
    text = "Deeply disappointed by Candidate B's economic proposal. Rising inflation and corruption allegations are alarming."
    res = analyzer.analyze(text)
    assert res["ml_prediction"]["sentiment"] == "Negative"
    assert res["vader_analysis"]["sentiment"] == "Negative"
    assert "Candidate B (NDU)" in res["entities"]

def test_neutral_sentiment(analyzer):
    text = "Election Update: Polling stations will open at 7:00 AM on election day across all districts."
    res = analyzer.analyze(text)
    assert res["ml_prediction"]["sentiment"] in ["Neutral", "Positive"]
    assert "preprocessing" in res

def test_empty_input(analyzer):
    res = analyzer.analyze("")
    assert "error" in res
