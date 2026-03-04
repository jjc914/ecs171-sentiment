import os
import re
import sys

import joblib

_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS = {0: "Negative", 1: "Positive"}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)   # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)  # punctuation / numbers
    text = re.sub(r"\s+", " ", text)        # whitespace
    return text.strip()


def load_nb_model():
    print("Loading Naive Bayes model...")

    # inject clean_text before loading pipeline
    sys.modules["__main__"].clean_text = clean_text

    pipeline = joblib.load(os.path.join(_DIR, "imdb_naive_bayes_pipeline.joblib"))
    
    print("Finished loading Naive Bayes model...")
    return {
        "pipeline": pipeline
    }

def get_nb_attributions(model_dict, text):
    """Log-likelihood ratio contribution per token: tfidf(w) * log(P(w|pos) / P(w|neg))."""
    pipeline = model_dict["pipeline"]

    # Detect steps by capability (duck typing) — more robust than isinstance
    # TF-IDF vectorizer: has get_feature_names_out + build_analyzer
    # NB classifier:     has feature_log_prob_
    vec = nb_clf = None
    for _, step in pipeline.steps:
        if hasattr(step, "get_feature_names_out") and hasattr(step, "build_analyzer"):
            vec = step
        if hasattr(step, "feature_log_prob_"):
            nb_clf = step
    if vec is None or nb_clf is None:
        return []

    cleaned = clean_text(text)
    tfidf_vec = vec.transform([cleaned])
    feature_names = vec.get_feature_names_out()

    word_scores = {}
    cx = tfidf_vec.tocsr()
    for feat_idx, tfidf_val in zip(cx.indices, cx.data):
        word = feature_names[feat_idx]
        log_ratio = float(nb_clf.feature_log_prob_[1, feat_idx] - nb_clf.feature_log_prob_[0, feat_idx])
        word_scores[word] = float(tfidf_val) * log_ratio

    tokens = vec.build_analyzer()(cleaned)
    return [{"token": t, "weight": word_scores.get(t, 0.0)} for t in tokens]


def infer_nb(model_dict, text):
    pipeline = model_dict["pipeline"]
    cleaned = clean_text(text)
    idx  = int(pipeline.predict([cleaned])[0])
    prob = float(pipeline.predict_proba([cleaned])[0][idx])

    return {
        "label":        LABELS[idx],
        "score":        round(prob, 4),
        "attributions": get_nb_attributions(model_dict, text),
    }
