
import numpy as np
import pytest

from app.lime_explainer import (
    explain_job_prediction,
    find_suspicious_class_index,
)


class DemoModel:
    """Small model used only for testing the LIME integration."""

    classes_ = np.array([0, 1])

    def predict_proba(self, texts):
        probabilities = []

        for text in texts:
            text = text.lower()

            score = sum(
                phrase in text
                for phrase in [
                    "registration fee",
                    "upfront payment",
                    "guaranteed income",
                ]
            )

            suspicious_probability = min(
                0.10 + (0.25 * score),
                0.95,
            )

            probabilities.append([
                1.0 - suspicious_probability,
                suspicious_probability,
            ])

        return np.array(probabilities)

    def predict(self, texts):
        probabilities = self.predict_proba(texts)
        return (probabilities[:, 1] >= 0.5).astype(int)


def test_find_suspicious_class_index():
    assert find_suspicious_class_index([0, 1]) == 1


def test_find_suspicious_class_by_name():
    assert find_suspicious_class_index(["genuine", "fraud"]) == 1


def test_unknown_suspicious_class_raises_error():
    with pytest.raises(ValueError):
        find_suspicious_class_index(["genuine", "legitimate"])


def test_lime_returns_explanation():
    model = DemoModel()

    result = explain_job_prediction(
        model=model,
        job_description=(
            "Urgent hiring for a developer. "
            "Pay a registration fee for guaranteed income."
        ),
        num_features=5,
        num_samples=100,
    )

    assert result["explanation_method"] == "LIME text explainer"
    assert result["suspicious_class"] == "1"
    assert 0 <= result["suspicious_class_probability_percent"] <= 100
    assert isinstance(result["supporting_features"], list)
    assert isinstance(result["lowering_features"], list)
    assert result["limitations"]


def test_lime_rejects_short_text():
    with pytest.raises(ValueError):
        explain_job_prediction(
            DemoModel(),
            "Too short",
            num_samples=100,
        )
