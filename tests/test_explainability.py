
from app.explainability import analyse_job


def test_detects_upfront_payment():
    result = analyse_job(
        "Please pay a registration fee to secure the job."
    )

    warning_ids = [item["id"] for item in result["warnings"]]

    assert "upfront_payment" in warning_ids


def test_detects_guaranteed_income():
    result = analyse_job(
        "Join our team today. Guaranteed income for every applicant."
    )

    warning_ids = [item["id"] for item in result["warnings"]]

    assert "unrealistic_guarantee" in warning_ids


def test_detects_sensitive_information_request():
    result = analyse_job(
        "Send your OTP to the recruiter to complete registration."
    )

    warning_ids = [item["id"] for item in result["warnings"]]

    assert "sensitive_information" in warning_ids


def test_no_warning_pattern_is_not_proof_of_safety():
    result = analyse_job(
        "We are hiring a Python developer. Apply through our careers team."
    )

    assert result["warning_count"] == 0
    assert "does not prove" in result["limitations"].lower()


def test_missing_employer_details_are_reported():
    result = analyse_job(
        "We are hiring a data analyst with Python and SQL experience."
    )

    missing_fields = [
        item["field"]
        for item in result["details_not_supplied"]
    ]

    assert "Company name" in missing_fields
    assert "Recruiter email" in missing_fields
    assert "Company website" in missing_fields


def test_supplied_employer_details_are_not_marked_missing():
    result = analyse_job(
        job_description=(
            "We are hiring a data analyst with Python and SQL experience."
        ),
        company_name="Example Technologies",
        recruiter_email="recruiter@example.com",
        company_website="https://example.com",
    )

    assert result["details_not_supplied"] == []


def test_no_fee_statement_does_not_trigger_payment_rule():
    result = analyse_job(
        "No registration fee is required. "
        "We are hiring a Python developer."
    )

    warning_ids = [item["id"] for item in result["warnings"]]

    assert "upfront_payment" not in warning_ids
