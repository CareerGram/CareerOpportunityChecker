
import re
import fitz  # PyMuPDF


# Skill names and common alternative spellings.
SKILL_PATTERNS = {
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b", r"\bstructured query language\b"],
    "Machine Learning": [
        r"\bmachine learning\b",
        r"\bml\b",
    ],
    "Deep Learning": [r"\bdeep learning\b"],
    "Data Analysis": [r"\bdata analysis\b", r"\bdata analytics\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "TensorFlow": [r"\btensorflow\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Scikit-learn": [r"\bscikit.learn\b", r"\bsklearn\b"],
    "Excel": [r"\bexcel\b", r"\bmicrosoft excel\b"],
    "Power BI": [r"\bpower bi\b"],
    "Tableau": [r"\btableau\b"],
    "Statistics": [r"\bstatistics\b", r"\bstatistical analysis\b"],
    "NLP": [r"\bnlp\b", r"\bnatural language processing\b"],
    "Computer Vision": [r"\bcomputer vision\b"],
    "FastAPI": [r"\bfastapi\b"],
    "Flask": [r"\bflask\b"],
    "Django": [r"\bdjango\b"],
    "Streamlit": [r"\bstreamlit\b"],
    "Git": [r"\bgit\b", r"\bgit version control\b"],
    "Docker": [r"\bdocker\b"],
    "AWS": [r"\baws\b", r"\bamazon web services\b"],
    "Communication": [r"\bcommunication skills\b"],
    "Project Management": [r"\bproject management\b"],
}


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF supplied as bytes."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    text_parts = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                raise ValueError("The PDF has no pages.")

            if document.page_count > 20:
                raise ValueError("The PDF must contain 20 pages or fewer.")

            for page in document:
                text_parts.append(page.get_text())

    except (fitz.FileDataError, RuntimeError) as exc:
        raise ValueError(
            "The uploaded file could not be read as a valid PDF."
        ) from exc

    text = "\n".join(text_parts).strip()

    if not text:
        raise ValueError(
            "No readable text was found. If this is a scanned resume, "
            "OCR support is needed in a future version."
        )

    return text


def find_skills(text: str) -> set[str]:
    """Return the known skills found in a piece of text."""

    normalised_text = text.lower()
    found = set()

    for skill, patterns in SKILL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalised_text):
                found.add(skill)
                break

    return found


def match_resume_to_job(
    resume_text: str,
    job_description: str,
) -> dict:
    """Compare known skills in a resume and a job description."""

    resume_skills = find_skills(resume_text)
    job_skills = find_skills(job_description)

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    extra = sorted(resume_skills - job_skills)

    if not job_skills:
        score = None
    else:
        score = round(
            len(matched) / len(job_skills) * 100,
            2,
        )

    return {
        "score": score,
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched,
        "missing_skills": missing,
        "additional_resume_skills": extra,
        "message": (
            "No recognised skills were found in the job description."
            if not job_skills
            else "Score is based only on the skills recognised by this tool."
        ),
    }
