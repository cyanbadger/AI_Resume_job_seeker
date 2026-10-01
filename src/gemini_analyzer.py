import os
import json
import time

from google import genai


def analyze_skill_gap(resume_text, job_description):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. "
            "Check your .env file."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are an expert technical recruiter and resume analyst.

Compare the candidate's resume with the job description.

================ RESUME ================
{resume_text}

================ JOB DESCRIPTION ================
{job_description}

Identify the important technical and professional skills
required by the job and compare them with the resume.

Classify each important skill as:

MATCHED:
Clearly demonstrated in the resume.

PARTIAL:
Related evidence exists, but the match is incomplete.

MISSING:
No reasonable evidence exists in the resume.

Rules:

1. Do not invent candidate experience.
2. Do not assume a skill exists because it is common.
3. Treat obvious synonyms as equivalent.
4. Focus on job-relevant skills.
5. Keep skill names concise.
6. Use evidence from the resume.
7. Return ONLY valid JSON.
8. Do not use Markdown code fences.

Return exactly:

{{
    "overall_summary": "Short summary of the candidate's alignment.",

    "required_skills": [
        {{
            "skill": "Python",
            "status": "MATCHED",
            "evidence": "Python is explicitly demonstrated in the resume."
        }}
    ],

    "matched_skills": [
        "Python"
    ],

    "partial_skills": [
        "REST APIs"
    ],

    "missing_skills": [
        "Docker"
    ],

    "recommendations": [
        "Add practical Docker experience through a relevant project."
    ]
}}
"""

    # --------------------------------------------------------
    # Models to try
    # --------------------------------------------------------

    models = [
        "gemini-3.8-flash",
        "gemini-2.5-flash"
    ]

    last_error = None

    # --------------------------------------------------------
    # Try each model
    # --------------------------------------------------------

    for model in models:

        # Retry temporary 503 errors
        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                response_text = response.text.strip()

                # Remove Markdown fences
                if response_text.startswith("```json"):
                    response_text = response_text[
                        len("```json"):
                    ]

                elif response_text.startswith("```"):
                    response_text = response_text[
                        len("```"):
                    ]

                if response_text.endswith("```"):
                    response_text = response_text[
                        :-len("```")
                    ]

                response_text = response_text.strip()

                result = json.loads(
                    response_text
                )

                return result

            except Exception as e:

                last_error = e

                error_text = str(e)

                # Retry only temporary service errors
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                ):

                    if attempt < 2:

                        time.sleep(
                            2 ** attempt
                        )

                        continue

                # Try next model
                break

    # --------------------------------------------------------
    # All attempts failed
    # --------------------------------------------------------

    raise RuntimeError(
        "Gemini is temporarily unavailable. "
        "Please try the analysis again in a few moments.\n\n"
        f"Last error: {last_error}"
    )