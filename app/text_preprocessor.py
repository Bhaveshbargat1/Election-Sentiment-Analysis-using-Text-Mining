"""
Text Preprocessor Module for Election Social Media Mining
Implements end-to-end NLP preprocessing: cleaning, contraction expansion,
tokenization, stop-word removal, and lemmatization.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Contraction mapping dictionary for text normalization
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "i've": "i have",
    "they've": "they have",
    "we've": "we have",
    "you've": "you have",
    "i'll": "i will",
    "they'll": "they will",
    "we'll": "we will",
    "you'll": "you will",
    "let's": "let us"
}

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))
            
        try:
            self.lemmatizer = WordNetLemmatizer()
            # test lemmatizer
            _ = self.lemmatizer.lemmatize("running")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            self.lemmatizer = WordNetLemmatizer()

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        lower_text = text.lower()
        for contraction, expansion in CONTRACTIONS.items():
            lower_text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, lower_text)
        return lower_text

    def clean_text(self, text: str) -> str:
        """
        Removes URLs, user mentions, preserves hashtag words, removes punctuation and digits.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        
        # 2. Remove user mentions (@username)
        text = re.sub(r"@\w+", "", text)
        
        # 3. Strip '#' from hashtags but keep the keyword (#Economy -> Economy)
        text = re.sub(r"#(\w+)", r"\1", text)
        
        # 4. Remove special characters and digits, keep alphabetic words
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        
        # 5. Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def tokenize(self, text: str) -> list:
        """Tokenizes cleaned text into lowercase word tokens."""
        return [word.lower() for word in text.split() if len(word) > 1]

    def remove_stopwords(self, tokens: list) -> list:
        """Filters out non-informative English stop words."""
        return [word for word in tokens if word not in self.stop_words]

    def lemmatize(self, tokens: list) -> list:
        """Reduces tokens to their base lemma."""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def transform(self, text: str) -> str:
        """
        Full end-to-end preprocessing pipeline returning space-separated lemma string.
        """
        expanded = self.expand_contractions(text)
        cleaned = self.clean_text(expanded)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(filtered)
        return " ".join(lemmas)

    def pipeline_step_by_step(self, text: str) -> dict:
        """
        Returns step-by-step transformation breakdown for UI demonstration and analysis.
        """
        expanded = self.expand_contractions(text)
        cleaned = self.clean_text(expanded)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        lemmas = self.lemmatize(filtered)
        final_text = " ".join(lemmas)
        
        return {
            "original": text,
            "expanded_contractions": expanded,
            "cleaned": cleaned,
            "tokens": tokens,
            "stopwords_removed": filtered,
            "lemmatized_tokens": lemmas,
            "processed_text": final_text
        }
