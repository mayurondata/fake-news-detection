"""
tests/test_predictor.py
────────────────────────
Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.predictor import preprocess, PredictionResult


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_lowercasing(self):
        result = preprocess("Hello World This Is A Test Article")
        assert result == result.lower()

    def test_url_removal(self):
        text = "Visit https://example.com for more info on this story"
        result = preprocess(text)
        assert "http" not in result
        assert "example" not in result

    def test_html_removal(self):
        text = "<p>Breaking news: <b>Scientists discover</b> water on Mars</p>"
        result = preprocess(text)
        assert "<" not in result
        assert ">" not in result

    def test_punctuation_removal(self):
        text = "Breaking! Scientists discover water on Mars, officials say."
        result = preprocess(text)
        assert "!" not in result
        assert "," not in result
        assert "." not in result

    def test_stopword_removal(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = preprocess(text)
        # 'the', 'over' are stopwords; 'quick', 'brown' etc. should remain
        tokens = result.split()
        assert "the" not in tokens

    def test_short_tokens_removed(self):
        # tokens ≤ 2 chars should be removed
        text = "a be to go big news story discovered today"
        result = preprocess(text)
        tokens = result.split()
        for t in tokens:
            assert len(t) > 2, f"Short token found: '{t}'"

    def test_empty_string(self):
        result = preprocess("")
        assert result == ""

    def test_only_noise(self):
        result = preprocess("https://example.com <b> !! ??? 123")
        # All content should be stripped
        assert len(result.strip()) == 0

    def test_returns_string(self):
        result = preprocess("Some valid news text about the economy")
        assert isinstance(result, str)

    def test_lemmatization(self):
        # 'running' → 'running' (nltk default lemmatizer uses noun POS by default)
        # We just check the pipeline doesn't crash and returns non-empty output
        result = preprocess("Scientists are discovering new species in the Amazon rainforest")
        assert len(result) > 0

    def test_numeric_removal(self):
        text = "The stock market fell 300 points on Tuesday amid inflation fears"
        result = preprocess(text)
        assert "300" not in result


# ──────────────────────────────────────────────────────────────────────────────
# PredictionResult dataclass tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictionResult:

    def _make_result(self, real_prob=0.85):
        fake_prob = 1.0 - real_prob
        label     = "REAL" if real_prob >= 0.5 else "FAKE"
        confidence = real_prob if label == "REAL" else fake_prob
        return PredictionResult(
            label             = label,
            confidence        = round(confidence, 4),
            real_prob         = round(real_prob, 4),
            fake_prob         = round(fake_prob, 4),
            word_count        = 100,
            clean_token_count = 60,
        )

    def test_real_label_when_prob_high(self):
        r = self._make_result(real_prob=0.9)
        assert r.label == "REAL"

    def test_fake_label_when_prob_low(self):
        r = self._make_result(real_prob=0.1)
        assert r.label == "FAKE"

    def test_probs_sum_to_one(self):
        r = self._make_result(real_prob=0.73)
        assert abs(r.real_prob + r.fake_prob - 1.0) < 1e-4

    def test_confidence_is_max_prob(self):
        r = self._make_result(real_prob=0.85)
        assert r.confidence == r.real_prob          # REAL prediction

        r2 = self._make_result(real_prob=0.15)
        assert r2.confidence == r2.fake_prob        # FAKE prediction

    def test_boundary_exactly_half(self):
        r = self._make_result(real_prob=0.5)
        assert r.label == "REAL"                    # 0.5 → REAL (≥ threshold)

    def test_word_count_stored(self):
        r = self._make_result()
        assert r.word_count == 100

    def test_token_count_stored(self):
        r = self._make_result()
        assert r.clean_token_count == 60


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing edge-case integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessIntegration:

    REAL_ARTICLE = """
    The Federal Reserve raised interest rates by a quarter percentage point on Wednesday,
    continuing its campaign to bring down inflation while signalling it may be nearing
    the end of its rate-hiking cycle. Fed Chair Jerome Powell said at a news conference
    that officials have seen encouraging progress on inflation but are not yet confident
    it is on a sustainable path back to the 2 percent target.
    """

    FAKE_ARTICLE = """
    BREAKING: Scientists at a top secret underground laboratory confirmed the existence
    of a massive cover-up involving world leaders and extraterrestrial beings. Anonymous
    insider sources claim the government has been hiding alien technology for decades.
    Share this before it gets taken down! The deep state cannot suppress this any longer.
    """

    def test_real_article_produces_tokens(self):
        result = preprocess(self.REAL_ARTICLE)
        tokens = result.split()
        assert len(tokens) > 10, "Expected many tokens from a real article"

    def test_fake_article_produces_tokens(self):
        result = preprocess(self.FAKE_ARTICLE)
        tokens = result.split()
        assert len(tokens) > 10

    def test_different_articles_produce_different_output(self):
        r1 = preprocess(self.REAL_ARTICLE)
        r2 = preprocess(self.FAKE_ARTICLE)
        assert r1 != r2

    def test_deterministic(self):
        """Same input must always produce the same output."""
        r1 = preprocess(self.REAL_ARTICLE)
        r2 = preprocess(self.REAL_ARTICLE)
        assert r1 == r2
