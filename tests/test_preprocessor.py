import pytest
from app.text_preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_expand_contractions(preprocessor):
    text = "They won't support this and can't believe it"
    res = preprocessor.expand_contractions(text)
    assert "will not" in res
    assert "cannot" in res

def test_clean_text_removes_urls_mentions(preprocessor):
    text = "Check out https://t.co/xyz123 and follow @CandidateA for #Economy updates!"
    res = preprocessor.clean_text(text)
    assert "https" not in res
    assert "@CandidateA" not in res
    assert "CandidateA" not in res
    assert "Economy" in res  # hashtag word preserved
    assert "#" not in res

def test_tokenization_and_stopwords(preprocessor):
    text = "the candidate has delivered an outstanding victory"
    tokens = preprocessor.tokenize(text)
    assert "candidate" in tokens
    assert "the" in tokens
    
    filtered = preprocessor.remove_stopwords(tokens)
    assert "the" not in filtered
    assert "candidate" in filtered
    assert "outstanding" in filtered

def test_lemmatization(preprocessor):
    tokens = ["promises", "running", "debates"]
    lemmas = preprocessor.lemmatize(tokens)
    assert "promise" in lemmas
    assert "debate" in lemmas

def test_pipeline_step_by_step(preprocessor):
    sample = "Don't miss @CandidateA's speech on #Healthcare! https://vote.com"
    steps = preprocessor.pipeline_step_by_step(sample)
    
    assert "original" in steps
    assert "expanded_contractions" in steps
    assert "cleaned" in steps
    assert "tokens" in steps
    assert "stopwords_removed" in steps
    assert "lemmatized_tokens" in steps
    assert "processed_text" in steps
    assert len(steps["lemmatized_tokens"]) > 0
