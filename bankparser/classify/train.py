"""
Train + evaluate the transaction classifier.

Evaluation is deliberately strict: models are trained on narrations of one set of merchants and
tested on narrations of DIFFERENT merchants (never seen in training), which mirrors reality.

Run:  python -m bankparser.classify.train
Writes models/classifier.joblib and models/metrics.json
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from .ml import DEFAULT_MODEL, MLModel
from .rules import RuleEngine

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "models" / "metrics.json"
FEEDBACK_WEIGHT = 5.0          # a user correction counts as much as 5 synthetic examples


def _synthetic(split, n, seed):
    from synthetic.narrations import make_dataset
    return make_dataset(n_per_category=n, seed=seed, split=split)


def _feedback_rows(path: Path | None):
    if not path or not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [(r["description"], r["direction"], r["category"]) for r in csv.DictReader(fh)]


def evaluate(n_train=300, n_test=80) -> dict:
    rules = RuleEngine()
    dirs = rules.categories
    train = _synthetic("train", n_train, seed=1)
    test = _synthetic("test", n_test, seed=2)
    tx, td, ty = zip(*train)
    ex, ed, ey = zip(*test)
    report = {"train_rows": len(train), "test_rows": len(test), "models": {}}

    for kind in ["logreg", "linear_svm", "naive_bayes", "random_forest"]:
        t0 = time.time()
        m = MLModel.train(tx, td, ty, dirs, kind=kind)
        pred = [p[0] for p in m.predict(ex, ed)]
        report["models"][kind] = {"accuracy": round(accuracy_score(ey, pred), 4),
                                  "macro_f1": round(f1_score(ey, pred, average="macro"), 4),
                                  "train_seconds": round(time.time() - t0, 1)}
        print(f"  {kind:14s} acc={report['models'][kind]['accuracy']:.3f} "
              f"macroF1={report['models'][kind]['macro_f1']:.3f}")

    # rules alone, and the full hybrid (rules first, then logistic regression)
    lr = MLModel.train(tx, td, ty, dirs, kind="logreg")
    ml_pred = lr.predict(ex, ed)
    rule_hits, rule_correct, hybrid = 0, 0, []
    for (text, d, y), (mp, prob, _) in zip(test, ml_pred):
        cat, _ = rules.match(text, d)
        if cat:
            rule_hits += 1
            rule_correct += cat == y
            hybrid.append(cat)
        else:
            hybrid.append(mp if prob >= 0.40 else "Uncategorized")
    report["rules"] = {"coverage": round(rule_hits / len(test), 4),
                       "precision": round(rule_correct / max(rule_hits, 1), 4)}
    report["hybrid"] = {"accuracy": round(accuracy_score(ey, hybrid), 4),
                        "macro_f1": round(f1_score(ey, hybrid, average="macro"), 4),
                        "uncategorized_rate": round(hybrid.count("Uncategorized") / len(hybrid), 4)}
    report["hybrid_per_class"] = classification_report(ey, hybrid, output_dict=True, zero_division=0)
    errors = Counter((y, p) for y, p in zip(ey, hybrid) if y != p)
    report["top_confusions"] = [f"{y} -> {p}: {c}" for (y, p), c in errors.most_common(8)]
    print(f"  rules: coverage={report['rules']['coverage']:.3f} precision={report['rules']['precision']:.3f}")
    print(f"  HYBRID acc={report['hybrid']['accuracy']:.3f} macroF1={report['hybrid']['macro_f1']:.3f} "
          f"uncategorized={report['hybrid']['uncategorized_rate']:.3f}")
    return report


def train_final_model(feedback_path=None, model_path=DEFAULT_MODEL, n=300) -> dict:
    """Train the production model on ALL synthetic merchants + user corrections."""
    rules = RuleEngine()
    rows = _synthetic("all", n, seed=3)
    fb = _feedback_rows(feedback_path)
    texts = [r[0] for r in rows] + [r[0] for r in fb]
    dirs = [r[1] for r in rows] + [r[1] for r in fb]
    labels = [r[2] for r in rows] + [r[2] for r in fb]
    weights = [1.0] * len(rows) + [FEEDBACK_WEIGHT] * len(fb)
    model = MLModel.train(texts, dirs, labels, rules.categories, kind="logreg", sample_weight=weights)
    model.save(model_path)
    return {"rows": len(rows), "feedback_rows": len(fb), "categories": len(set(labels))}


def main():
    from .hybrid import DEFAULT_FEEDBACK
    print("Evaluating on held-out (unseen) merchants...")
    report = evaluate()
    info = train_final_model(feedback_path=DEFAULT_FEEDBACK)
    report["final_model"] = info
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model -> {DEFAULT_MODEL}  ({info})")
    print("Top confusions:", *report["top_confusions"], sep="\n  ")


if __name__ == "__main__":
    main()
