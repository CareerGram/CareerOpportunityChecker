
from pathlib import Path

import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.resume_matcher import extract_pdf_text, match_resume_to_job
from app.explainability import analyse_job
from app.lime_explainer import explain_job_prediction



PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

app = FastAPI(
    title="Career Opportunity Checker API",
    version="2.0.0",
)

@app.get("/")
def root():
    return {
        "message": "Welcome to Career Opportunity Checker API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }

model = None


@app.on_event("startup")
def load_model():
    """Load the trained Phase 1 model when the API starts."""
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run the Phase 1 training script first."
        )

    model = joblib.load(MODEL_PATH)


class JobRequest(BaseModel):
    job_description: str = Field(
        min_length=20,
        max_length=50000,
        description="Job title, description, requirements and related text",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_job(request: JobRequest):
    """Run the Phase 1 ML model on the supplied job text."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded.",
        )

    try:
        prediction = model.predict([request.job_description])[0]
        label = str(prediction)

        # This assumes the Phase 1 target uses 1 for fraudulent
        # and 0 for genuine. Check the dataset's target encoding.
        if prediction == 1 or label.lower() in {
            "fake", "fraudulent", "fraud", "scam"
        }:
            risk = "Potentially suspicious"
        elif prediction == 0 or label.lower() in {
            "real", "genuine", "legitimate"
        }:
            risk = "Lower predicted risk"
        else:
            risk = "Unclassified prediction"

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                [request.job_description]
            )[0]

            classes = list(model.classes_)
            suspicious_indices = [
                i for i, class_label in enumerate(classes)
                if (
                    class_label == 1
                    or str(class_label).lower() in {
                        "fake", "fraudulent", "fraud", "scam"
                    }
                )
            ]

            if suspicious_indices:
                probability = round(
                    float(probabilities[suspicious_indices[0]]) * 100,
                    2,
                )

        return {
            "prediction": label,
            "risk_assessment": risk,
            "suspicious_class_probability_percent": probability,
            "warning": (
                "This is a model prediction, not proof that a job is "
                "fraudulent or genuine. Verify the employer independently."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/resume-match")
async def resume_match(
    job_description: str = Form(..., min_length=20, max_length=50000),
    resume: UploadFile = File(...),
):
    """Extract a PDF resume and compare its skills with a job."""

    if resume.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF resume.",
        )

    pdf_bytes = await resume.read()

    # Limit the upload to 5 MB.
    if len(pdf_bytes) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The resume PDF must be 5 MB or smaller.",
        )

    try:
        resume_text = extract_pdf_text(pdf_bytes)
        result = match_resume_to_job(
            resume_text,
            job_description,
        )
        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        await resume.close()



class ExplainRequest(BaseModel):
    job_description: str = Field(
        min_length=20,
        max_length=50000,
    )
    company_name: str = Field(default="", max_length=200)
    recruiter_email: str = Field(default="", max_length=320)
    company_website: str = Field(default="", max_length=500)


@app.post("/explain")
def explain_job(request: ExplainRequest):
    """Return rule-based warning indicators and verification advice."""

    return analyse_job(
        job_description=request.job_description,
        company_name=request.company_name,
        recruiter_email=request.recruiter_email,
        company_website=request.company_website,
    )


@app.post("/explain-model")
def explain_model(request: JobRequest):
    """Explain the existing model's prediction using LIME."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The ML model is not loaded.",
        )

    try:
        return explain_job_prediction(
            model=model,
            job_description=request.job_description,
            num_features=10,
            num_samples=1000,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "The model explanation could not be generated. "
                "Check the saved model and its input format."
            ),
        ) from exc
