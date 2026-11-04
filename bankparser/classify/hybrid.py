"""
Hybrid classifier = 3 layers, each explainable:

  1. USER MEMORY  - corrections the user made in the UI (exact merchant/person key)  -> 100% trust
  2. RULES        - keyword / regex rules from config/categories.yaml                -> high precision
  3. ML MODEL     - TF-IDF + Logistic Regression, for anything rules don't cover     -> generalises
  If the ML confidence is below the threshold the row is "Uncategorized" and flagged for review,
  instead of silently guessing.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from .ml import DEFAULT_MODEL, MLModel
from .rules import DEFAULT_RULES, RuleEngine
from .text import memory_key

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEEDBACK = ROOT / "data" / "feedback" / "corrections.csv"


class HybridClassifier:
    def __init__(self, rules_path=DEFAULT_RULES, model_path=DEFAULT_MODEL, feedback_path=DEFAULT_FEEDBACK,
                 threshold: float = 0.40):
        self.rules = RuleEngine(rules_path)
        self.model_path = Path(model_path)
        self.model = MLModel.load(model_path) if Path(model_path).exists() else None
        self.feedback_path = Path(feedback_path)
        self.threshold = threshold
        self.memory: dict[str, str] = {}
        self._load_feedback()

    @property
    def categories(self) -> list[str]:
        return list(self.rules.categories)

    # ------------------------------------------------ feedback loop
    def _load_feedback(self):
        if not self.feedback_path.exists():
            return
        with open(self.feedback_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.memory[memory_key(row["description"], row["direction"])] = row["category"]

    def add_feedback(self, description: str, direction: str, category: str):
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.feedback_path.exists()
        with open(self.feedback_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["timestamp", "description", "direction", "category"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), description, direction, category])
        self.memory[memory_key(description, direction)] = category

    def retrain(self):
        from .train import train_final_model
        metrics = train_final_model(feedback_path=self.feedback_path, model_path=self.model_path)
        self.model = MLModel.load(self.model_path)
        return metrics

    # ------------------------------------------------ classification
    def classify(self, description: str, direction: str):
        """Returns (category, source, confidence, reason)."""
        key = memory_key(description, direction)
        if key in self.memory:
            return self.memory[key], "user", 1.0, "learned from your earlier correction"
        cat, why = self.rules.match(description, direction)
        if cat:
            return cat, "rule", 1.0, f"rule: {why}"
        if self.model is not None:
            best, p, top = self.model.predict([description], [direction])[0]
            alts = ", ".join(f"{c} {q:.0%}" for c, q in top)
            if p >= self.threshold:
                return best, "ml", round(p, 3), f"ML model ({alts})"
            return "Uncategorized", "none", round(p, 3), f"low ML confidence - review ({alts})"
        return "Uncategorized", "none", 0.0, "no rule matched and no ML model loaded"

    def classify_all(self, txns):
        pending = []
        for t in txns:
            direction = "credit" if t.credit > 0 and not t.debit else "debit"
            key = memory_key(t.description, direction)
            if key in self.memory:
                t.category, t.category_source, t.confidence, t.category_reason = \
                    self.memory[key], "user", 1.0, "learned from your earlier correction"
                continue
            cat, why = self.rules.match(t.description, direction)
            if cat:
                t.category, t.category_source, t.confidence, t.category_reason = cat, "rule", 1.0, f"rule: {why}"
            else:
                pending.append((t, direction))
        if pending and self.model is not None:       # one batched ML call for speed
            preds = self.model.predict([t.description for t, _ in pending], [d for _, d in pending])
            for (t, _), (best, p, top) in zip(pending, preds):
                alts = ", ".join(f"{c} {q:.0%}" for c, q in top)
                if p >= self.threshold:
                    t.category, t.category_source, t.confidence, t.category_reason = best, "ml", round(p, 3), f"ML model ({alts})"
                else:
                    t.category, t.category_source, t.confidence, t.category_reason = \
                        "Uncategorized", "none", round(p, 3), f"low ML confidence - review ({alts})"
        elif pending:
            for t, _ in pending:
                t.category, t.category_source, t.confidence, t.category_reason = "Uncategorized", "none", 0.0, "no rule matched"
        return txns
