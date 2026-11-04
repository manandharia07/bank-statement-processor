"""
Traditional ML layer: TF-IDF features (word 1-2 grams + character 3-5 grams) -> Logistic Regression.

Character n-grams are the key choice: they recognise merchant names inside joined /
truncated / misspelt tokens ("ZOMATO12", "SWIGGYINSTA", "AMZN") that exact keywords miss.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .text import ml_text

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "classifier.joblib"


def build_pipeline(kind: str = "logreg") -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False,
                                 token_pattern=r"[A-Z_]{2,}")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                 lowercase=False, min_df=2)),
    ])
    models = {
        "logreg": LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced"),
        "linear_svm": LinearSVC(C=0.8, class_weight="balanced"),
        "naive_bayes": ComplementNB(alpha=0.3),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0,
                                                class_weight="balanced"),
    }
    return Pipeline([("features", features), ("clf", models[kind])])


class MLModel:
    def __init__(self, pipeline: Pipeline, category_directions: dict):
        self.pipeline = pipeline
        self.category_directions = category_directions

    @classmethod
    def train(cls, texts, directions, labels, category_directions, kind="logreg", sample_weight=None):
        pipe = build_pipeline(kind)
        X = [ml_text(t, d) for t, d in zip(texts, directions)]
        pipe.fit(X, labels, clf__sample_weight=sample_weight)
        return cls(pipe, category_directions)

    def save(self, path: Path | str = DEFAULT_MODEL):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "category_directions": self.category_directions}, path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL):
        obj = joblib.load(path)
        return cls(obj["pipeline"], obj["category_directions"])

    def predict(self, texts, directions):
        """Returns list of (best_category, probability, top3) with the direction constraint applied."""
        X = [ml_text(t, d) for t, d in zip(texts, directions)]
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            P = self.pipeline.predict_proba(X)
        else:  # LinearSVC: softmax over decision scores (only used for model comparison)
            s = self.pipeline.decision_function(X)
            P = np.exp(s - s.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
        classes = list(clf.classes_)
        out = []
        for row, d in zip(P, directions):
            mask = np.array([self.category_directions.get(c, "any") in (d, "any") for c in classes], dtype=float)
            p = row * mask
            p = p / p.sum() if p.sum() > 0 else row
            order = np.argsort(-p)[:3]
            out.append((classes[order[0]], float(p[order[0]]), [(classes[i], round(float(p[i]), 3)) for i in order]))
        return out
