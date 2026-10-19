
import fitz
import pytest

from app.resume_matcher import (
    extract_pdf_text,
    find_skills,
    match_resume_to_job,
)


def test_find_skills_recognises_aliases():
    text = "I use Python, SQL and scikit-learn."
    skills = find_skills(text)

    assert "Python" in skills
    assert "SQL" in skills
    assert "Scikit-learn" in skills


def test_resume_match_score():
    resume = "Experience with Python and SQL."
    job = "Requirements: Python, SQL and Tableau."

    result = match_resume_to_job(resume, job)

    assert result["score"] == 66.67
    assert result["matched_skills"] == ["Python", "SQL"]
    assert result["missing_skills"] == ["Tableau"]


def test_no_recognised_job_skills():
    result = match_resume_to_job(
        "I enjoy learning new things.",
        "We are looking for a motivated team member.",
    )

    assert result["score"] is None
    assert result["job_skills"] == []


def test_empty_pdf_is_rejected():
    with pytest.raises(ValueError):
        extract_pdf_text(b"")


def test_extract_pdf_text():
    # Create a small PDF in memory for testing.
    pdf_document = fitz.open()
    page = pdf_document.new_page()
    page.insert_text((72, 72), "Python SQL resume")
    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()

    text = extract_pdf_text(pdf_bytes)

    assert "Python" in text
    assert "SQL" in text
