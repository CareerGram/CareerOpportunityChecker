
import re


# Each rule explains a possible warning sign.
# These rules are indicators, not proof of fraud.
WARNING_RULES = [
    {
        "id": "upfront_payment",
        "title": "Request for upfront payment",
        "severity": "high",
        "patterns": [
            r"\bpay (?:an? )?(?:registration|application|processing|training|security) fee\b",
            r"\b(?:registration|application|processing|training|security) fee of\b",
            r"\bupfront payment\b",
            r"\bpay (?:us|the company|the recruiter) to get (?:the )?job\b",
            r"\bsend money to (?:get|secure|confirm) (?:the )?job\b",
        ],
        "explanation": (
            "The text appears to request money in connection with getting "
            "the job. Verify this carefully before paying anything."
        ),
        "advice": (
            "Contact the employer through independently verified official "
            "contact details. Do not pay to secure a job."
        ),
    },
    {
        "id": "unrealistic_guarantee",
        "title": "Guaranteed income or job claims",
        "severity": "medium",
        "patterns": [
            r"\bguaranteed (?:job|income|salary|earnings)\b",
            r"\bearn (?:rs\.?|inr|₹|\$)?\s*\d[\d,]*(?:\s*(?:per|a)\s*(?:day|week|month))?.{0,35}\bguaranteed\b",
            r"\b100\s*%\s*(?:job|placement|selection)\s*guarantee\b",
        ],
        "explanation": (
            "The posting appears to promise guaranteed employment or income. "
            "Such claims deserve verification, especially when combined "
            "with payment requests."
        ),
        "advice": (
            "Ask for written details of the role, compensation, and any "
            "conditions attached to the claim."
        ),
    },
    {
        "id": "pressure_tactics",
        "title": "Pressure to act immediately",
        "severity": "medium",
        "patterns": [
            r"\bapply immediately or lose\b",
            r"\blimited (?:time|slots|vacancies)\b",
            r"\bpay today\b",
            r"\bact now\b",
            r"\burgent hiring\b",
            r"\bimmediate payment required\b",
        ],
        "explanation": (
            "The posting uses urgency or limited-time language. Legitimate "
            "recruitment can also have deadlines, so this is not proof of fraud."
        ),
        "advice": (
            "Take time to verify the employer and job details before sharing "
            "money or sensitive information."
        ),
    },
    {
        "id": "sensitive_information",
        "title": "Potentially sensitive information request",
        "severity": "high",
        "patterns": [
            r"\b(?:send|share|provide|reveal|disclose)\s+(?:your\s+)?(?:bank password|net banking password)\b",
            r"\b(?:send|share|provide|reveal|disclose)\s+(?:your\s+)?(?:otp|one.time password|upi pin|pin number)\b",
            r"\b(?:otp|one.time password|upi pin|pin number)\s+(?:is required|must be shared|must be provided)\b",
            r"\bprovide\s+(?:your\s+)?(?:bank password|upi pin)\b",
        ],
        "explanation": (
            "The text appears to ask for highly sensitive banking or "
            "authentication information."
        ),
        "advice": (
            "Never share passwords, one-time passwords, or UPI PINs with "
            "a recruiter. Verify the message independently."
        ),
    },
]


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Return True when any warning pattern matches the text."""

    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        for pattern in patterns
    )


def analyse_job(
    job_description: str,
    company_name: str = "",
    recruiter_email: str = "",
    company_website: str = "",
) -> dict:
    """
    Find possible warning signs and identify details not supplied
    in the provided information.
    """

    text = job_description.strip()
    warnings = []

    # Avoid flagging common explicit statements that say fees are not required.
    # Remove these phrases before checking payment-related rules.
    payment_check_text = re.sub(
        r"\b(?:no|zero|without)\s+(?:registration|application|processing|"
        r"training|security)\s+fees?\s+(?:are\s+)?(?:required|needed|payable)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    payment_check_text = re.sub(
        r"\bno payment required\b|\bno fees? required\b",
        "",
        payment_check_text,
        flags=re.IGNORECASE,
    )

    for rule in WARNING_RULES:
        text_to_check = (
            payment_check_text
            if rule["id"] == "upfront_payment"
            else text
        )

        if _matches_any_pattern(text_to_check, rule["patterns"]):
            warnings.append({
                "id": rule["id"],
                "title": rule["title"],
                "severity": rule["severity"],
                "explanation": rule["explanation"],
                "advice": rule["advice"],
            })

    # These checks only tell us which details were supplied.
    # They do not verify that the details are genuine.
    missing_details = []

    if not company_name.strip():
        missing_details.append({
            "field": "Company name",
            "message": "No company name was supplied for checking.",
        })

    if not recruiter_email.strip():
        missing_details.append({
            "field": "Recruiter email",
            "message": "No recruiter email was supplied for checking.",
        })

    if not company_website.strip():
        missing_details.append({
            "field": "Company website",
            "message": "No company website was supplied for checking.",
        })

    if warnings:
        summary = (
            f"Detected {len(warnings)} type(s) of potential warning signs. "
            "Review them and verify the employer independently."
        )
    else:
        summary = (
            "No warning patterns from this rule set were detected. "
            "This does not mean the job is genuine or safe."
        )

    return {
        "summary": summary,
        "warning_count": len(warnings),
        "warnings": warnings,
        "details_not_supplied": missing_details,
        "verification_checklist": [
            "Find the company website independently; do not rely only on a link in the message.",
            "Check whether the job appears on the company's official careers page.",
            "Verify the recruiter using contact details published by the company.",
            "Do not pay fees to secure a job or share passwords, OTPs, or UPI PINs.",
            "Confirm the role, salary, location, and employment terms.",
        ],
        "limitations": (
            "This is a rule-based screening aid. It does not verify a company, "
            "website, email address, or job posting against external records. "
            "A warning sign is not proof of fraud, and no detected warnings "
            "does not prove that a job is genuine."
        ),
    }
