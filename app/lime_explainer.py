
"""Explain individual job predictions with LIME."""

import re

from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


SUSPICIOUS_LABELS = {
    "fake",
    "fraud",
    "fraudulent",
    "scam",
    "suspicious",
}

# These words can change the meaning of a sentence, so keep them.
NEGATION_WORDS = {"no", "not", "never", "nor", "none", "without"}


def find_suspicious_class_index(classes):
    """Find the model class that represents a suspicious job."""

    # Prefer a clearly named suspicious class when one is available.
    for index, label in enumerate(classes):
        if str(label).strip().lower() in SUSPICIOUS_LABELS:
            return index

    # Many binary classifiers use 1 for the suspicious class.
    # Check your dataset's label mapping before relying on this fallback.
    for index, label in enumerate(classes):
        if label == 1 or str(label).strip() == "1":
            return index

    raise ValueError(
        "Could not identify the suspicious class. "
        "Check the model's class labels and training label mapping."
    )


def is_unhelpful_feature(feature):
    """Return True when a feature contains only stop words."""

    words = re.findall(r"[a-zA-Z0-9_'-]+", feature.lower())

    if not words:
        return True

    meaningful_words = [
        word
        for word in words
        if word not in ENGLISH_STOP_WORDS or word in NEGATION_WORDS
    ]

    return not meaningful_words


def describe_feature(feature, weight):
    """Give a simple explanation of a feature's possible meaning."""

    text = feature.lower()

    if any(word in text for word in ("registration fee", "processing fee",
                                     "payment", "pay", "fee", "deposit")):
        return (
            "The text mentions a payment or fee. Verify the employer and "
            "the reason for the payment before proceeding."
        )

    if any(word in text for word in ("guaranteed selection", "guaranteed job",
                                     "guaranteed", "100% selection")):
        return (
            "A promise of guaranteed selection deserves careful checking. "
            "This wording alone does not prove that the job is fraudulent."
        )

    if any(word in text for word in ("urgent", "immediately", "limited time",
                                     "expires", "within 30 minutes")):
        return (
            "Urgent wording may pressure applicants to act quickly. "
            "Take time to verify the opportunity independently."
        )

    if any(word in text for word in ("bank details", "aadhaar", "passport",
                                     "identity proof", "personal documents")):
        return (
            "The text refers to personal or financial information. "
            "Verify the recipient and the purpose before sharing anything."
        )

    if any(word in text for word in ("no experience", "work from home",
                                     "high salary", "easy money")):
        return (
            "This wording can appear in both genuine and suspicious job ads. "
            "Check the full offer rather than judging by this phrase alone."
        )

    if any(word in text for word in ("interview", "technical assessment",
                                     "careers portal", "official website")):
        return (
            "This wording describes a recruitment process or application "
            "channel. It does not, by itself, prove that the employer is genuine."
        )

    if weight > 0:
        return (
            "LIME found that this text feature supports the model's "
            "suspicious-job prediction for this example."
        )

    return (
        "LIME found that this text feature moves the model away from "
        "the suspicious-job prediction for this example."
    )


def explain_job_prediction(
    model,
    job_description,
    num_features=10,
    num_samples=1000,
):
    """Explain one job prediction using a local LIME explanation."""

    if not isinstance(job_description, str) or len(job_description.strip()) < 20:
        raise ValueError(
            "Please provide a job description containing at least 20 characters."
        )

    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ValueError(
            "The model must provide predict_proba() and classes_."
        )

    classes = list(model.classes_)
    suspicious_index = find_suspicious_class_index(classes)

    probabilities = model.predict_proba([job_description])[0]
    suspicious_probability = float(probabilities[suspicious_index]) * 100
    predicted_label = model.predict([job_description])[0]

    explainer = LimeTextExplainer(
        class_names=[str(label) for label in classes],
        bow=True,
    )

    # Ask LIME for extra features because some will be filtered out.
    explanation = explainer.explain_instance(
        text_instance=job_description,
        classifier_fn=model.predict_proba,
        labels=[suspicious_index],
        num_features=max(num_features * 3, 30),
        num_samples=num_samples,
    )

    raw_features = explanation.as_list(label=suspicious_index)

    # Remove features that contain only stop words, then rank by weight.
    useful_features = [
        (feature, float(weight))
        for feature, weight in raw_features
        if not is_unhelpful_feature(feature)
    ]
    useful_features.sort(key=lambda item: abs(item[1]), reverse=True)
    useful_features = useful_features[:num_features]

    supporting_features = []
    lowering_features = []

    for feature, weight in useful_features:
        item = {
            "feature": feature,
            "weight": weight,
            "interpretation": (
                "Supports the suspicious-job prediction"
                if weight > 0
                else "Moves the prediction away from suspicious"
            ),
            "plain_english": describe_feature(feature, weight),
        }

        if weight > 0:
            supporting_features.append(item)
        elif weight < 0:
            lowering_features.append(item)

    return {
        "prediction": str(predicted_label),
        "suspicious_class": str(classes[suspicious_index]),
        "suspicious_class_probability_percent": round(
            suspicious_probability, 2
        ),
        "supporting_features": supporting_features,
        "lowering_features": lowering_features,
        "explanation_method": "LIME text explainer",
        "num_samples": num_samples,
        "limitations": [
            "LIME explains this individual prediction using a local approximation.",
            "Feature weights are not fraud probabilities or proof of fraud.",
            "Stop-word filtering cleans the displayed explanation; it does not "
            "remove those words from the model's original input.",
            "A job should be checked using independent evidence before making "
            "a final decision.",
        ],
    }
