import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Career Opportunity Checker",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Career Opportunity Checker")

st.write(
    "Screen job postings, understand ML predictions, and compare "
    "job requirements with your resume."
)

st.warning(
    "This tool provides screening assistance, not proof of fraud. "
    "Verify employers independently before making decisions."
)

job_description = st.text_area(
    "Job description",
    height=250,
    placeholder="Paste the complete job description here...",
)

st.subheader("Employer details")

company_name = st.text_input("Company name")
recruiter_email = st.text_input("Recruiter email")
company_website = st.text_input("Company website")

resume_file = st.file_uploader(
    "Resume PDF (optional, maximum 5 MB)",
    type=["pdf"],
)

if st.button("Analyse Job", type="primary"):
    if len(job_description.strip()) < 20:
        st.error("Enter a job description of at least 20 characters.")

    else:
        # 1. Existing ML prediction
        st.header("1. ML Job Risk Assessment")

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"job_description": job_description},
                timeout=60,
            )
            response.raise_for_status()
            prediction = response.json()

            st.write(
                "**Model assessment:**",
                prediction["risk_assessment"],
            )

            probability = prediction.get(
                "suspicious_class_probability_percent"
            )

            if probability is not None:
                st.metric(
                    "Model probability for the suspicious class",
                    f"{probability:.2f}%",
                )

            st.caption(prediction["warning"])

        except requests.RequestException as exc:
            st.error("ML prediction failed. Is FastAPI running?")
            st.caption(str(exc))

        # 2. New Phase 4 model explainability
        # 2. Explain why the model made its prediction
        st.header("2. Why Did the Model Make This Prediction?")

        try:
            response = requests.post(
                f"{API_URL}/explain-model",
                json={"job_description": job_description},
                timeout=120,
            )
            response.raise_for_status()
            explanation = response.json()

            suspicious_class = explanation["suspicious_class"]
            probability = explanation[
                "suspicious_class_probability_percent"
            ]

            st.subheader("A. Model output")

            st.write(
                f"The model's suspicious class is **{suspicious_class}**."
            )

            st.metric(
                "Model probability for the suspicious class",
                f"{probability:.2f}%",
            )

            st.caption(
                "This is the model's estimated class probability, "
                "not the verified probability that the job is fraudulent."
            )

            st.subheader("B. Which text features influenced the result?")

            supporting = explanation["supporting_features"]
            lowering = explanation["lowering_features"]

            all_features = supporting + lowering

            if all_features:
                # Show a chart of LIME's signed local weights.
                chart_data = {
                    item["feature"]: item["weight"]
                    for item in all_features
                }

                st.write(
                    "Positive weights support the suspicious class. "
                    "Negative weights lower its score in this explanation."
                )

                st.bar_chart(chart_data)

                st.caption(
                    "The bar values are LIME weights, not percentages. "
                    "Their magnitude is not the model's probability change."
                )
            else:
                st.info(
                    "LIME did not return non-zero feature weights for this input."
                )

            left, right = st.columns(2)

            with left:
                st.markdown("#### Features supporting suspicious")

                if supporting:
                    for index, item in enumerate(supporting, start=1):
                        with st.expander(
                            f'{index}. {item["feature"]} '
                            f'({item["weight"]:+.4f})'
                        ):
                            st.write(
                                "**What the weight means:** "
                                "This feature supports the suspicious class "
                                "in LIME's local approximation."
                            )
                            st.write(
                                "**General context:**",
                                item["plain_english"],
                            )
                            st.caption(item["strength"])
                else:
                    st.write(
                        "No positive feature weights were returned."
                    )

            with right:
                st.markdown("#### Features lowering suspicious")

                if lowering:
                    for index, item in enumerate(lowering, start=1):
                        with st.expander(
                            f'{index}. {item["feature"]} '
                            f'({item["weight"]:+.4f})'
                        ):
                            st.write(
                                "**What the weight means:** "
                                "This feature lowers the suspicious-class "
                                "score in LIME's local approximation."
                            )
                            st.write(
                                "**General context:**",
                                item["plain_english"],
                            )
                            st.caption(item["strength"])
                else:
                    st.write(
                        "No negative feature weights were returned."
                    )

            st.subheader("C. What should you do next?")

            st.markdown(
                "- Verify the employer using an independently found "
                "official website or contact number.\n"
                "- Never pay a recruitment fee just because a job posting "
                "requests it.\n"
                "- Avoid sharing bank details or identity documents "
                "before verifying the recipient and purpose.\n"
                "- Treat the model output as one screening signal, "
                "not a final decision."
            )

            st.subheader("D. Limitations")

            st.info(explanation["limitations"])

            st.caption(
                f'Explanation method: {explanation["explanation_method"]}; '
                f'LIME samples: {explanation["num_samples"]}.'
            )

        except requests.RequestException as exc:
            st.error(
                "LIME explanation failed. Check that FastAPI is running "
                "and that /explain-model is available."
            )
            st.caption(str(exc))

        except (KeyError, TypeError, ValueError) as exc:
            st.error(
                "The explanation response was incomplete or invalid. "
                "Check the API response and model class mapping."
            )
            st.caption(str(exc))


        # 3. Existing Phase 3 rule-based warning indicators
        st.header("3. Rule-Based Warning Indicators")

        try:
            response = requests.post(
                f"{API_URL}/explain",
                json={
                    "job_description": job_description,
                    "company_name": company_name,
                    "recruiter_email": recruiter_email,
                    "company_website": company_website,
                },
                timeout=30,
            )
            response.raise_for_status()
            rules = response.json()

            st.write(rules["summary"])

            if rules["warnings"]:
                for warning in rules["warnings"]:
                    with st.expander(
                        f'{warning["title"]} — '
                        f'{warning["severity"].upper()}'
                    ):
                        st.write(warning["explanation"])
                        st.markdown("**Recommended action**")
                        st.write(warning["advice"])
            else:
                st.info(
                    "No warning patterns were detected. This does not "
                    "prove that the job is genuine."
                )

            if rules["details_not_supplied"]:
                st.markdown("**Employer details not supplied**")
                for item in rules["details_not_supplied"]:
                    st.write(
                        f'{item["field"]}: {item["message"]}'
                    )

            st.markdown("**Verification checklist**")
            for item in rules["verification_checklist"]:
                st.checkbox(item, key=f"verify_{item}")

            st.caption(rules["limitations"])

        except requests.RequestException as exc:
            st.error("Rule-based analysis failed.")
            st.caption(str(exc))

        # 4. Existing Phase 2 resume matching
        st.header("4. Resume–Job Skill Match")

        if resume_file is None:
            st.info("Upload a resume PDF to calculate skill overlap.")

        else:
            try:
                response = requests.post(
                    f"{API_URL}/resume-match",
                    files={
                        "resume": (
                            resume_file.name,
                            resume_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    data={
                        "job_description": job_description,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                result = response.json()

                if result["score"] is None:
                    st.info(result["message"])
                else:
                    st.metric(
                        "Recognised skill overlap",
                        f'{result["score"]:.2f}%',
                    )
                    st.progress(int(result["score"]))

                left, right = st.columns(2)

                with left:
                    st.markdown("**Matching skills**")
                    st.write(
                        ", ".join(result["matched_skills"]) or "None found"
                    )

                with right:
                    st.markdown("**Job skills missing from resume**")
                    st.write(
                        ", ".join(result["missing_skills"]) or "None found"
                    )

                with st.expander("View all recognised skills"):
                    st.write("Resume skills:", result["resume_skills"])
                    st.write("Job skills:", result["job_skills"])
                    st.write(
                        "Additional resume skills:",
                        result["additional_resume_skills"],
                    )

                st.caption(result["message"])

            except requests.RequestException as exc:
                st.error("Resume matching failed.")
                st.caption(str(exc))

st.divider()
st.caption("Built by Team CareerGram")

