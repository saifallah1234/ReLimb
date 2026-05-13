from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groq import Groq

from src.core.settings import settings


def build_llm_prompt(label: str, notes: str, user_input: str = "") -> str:
    label_context = {
        "Normal Gait": "The patient's gait pattern appears normal with no significant biomechanical issues detected.",
        "Foot & Ankle Issue": (
            "The analysis detected irregularities in the foot and ankle region, suggesting possible instability, "
            "weakness, or compensatory movement patterns."
        ),
        "Knee Issue": (
            "The analysis detected abnormal movement patterns around the knee joint, which may indicate weakness, "
            "instability, or misalignment."
        ),
        "Alignment Issue": "The analysis detected postural or skeletal alignment issues affecting the overall gait pattern.",
        "Unknown / Other": "The analysis detected an irregular gait pattern that does not clearly match known categories.",
    }

    severity_map = {
        "Normal Gait": "None",
        "Foot & Ankle Issue": "Moderate",
        "Knee Issue": "Moderate",
        "Alignment Issue": "Moderate",
        "Unknown / Other": "Unclear — further evaluation needed",
    }

    label_description = label_context.get(label, "An irregular gait pattern was detected.")
    severity = severity_map.get(label, "Unknown")

    prompt = f"""
You are a compassionate and professional medical assistant specializing in physical rehabilitation and gait analysis.

Your role is to explain a patient's gait analysis result in a clear, reassuring, and non-alarming way.
The patient has no medical background, so avoid technical jargon. Be empathetic and supportive.

---

ANALYSIS RESULT:
- Detected Condition: {label}
- Clinical Description: {label_description}
- Severity: {severity}
- System Observations / Notes: {notes}

---

LANGUAGE INSTRUCTION:
Detect the language from this user input and respond in that exact language: "{user_input}"
If no user input is provided or language is unclear, respond in English.

---

RESPONSE FORMAT:
Respond ONLY in this exact structure (translate the keys too if needed):

**📋 What We Found:**
[Explain what was detected in 2-3 simple sentences. No medical jargon. Be calm and reassuring.]

**⚠️ Severity:**
[One of: None / Mild / Moderate / Severe — with one sentence explaining what that means for the patient.]

**✅ What You Should Do Next:**
[3 to 4 actionable bullet points the patient can realistically follow. Include when to see a doctor if needed.]

**💬 Encouragement:**
[One short warm sentence to reassure and motivate the patient.]

---

STRICT RULES:
- Never diagnose. You are explaining a screening result, not providing a medical diagnosis.
- Never use terms like "disease", "disorder", "pathology" unless absolutely necessary.
- Always recommend consulting a healthcare professional for confirmation.
- Keep the tone warm, clear, and human.
- Do not add anything outside the format above.
"""
    return prompt.strip()


def _get_client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured. Set it in .env or environment variables.")
    return Groq(api_key=settings.groq_api_key)


def generate_llm_summary(label: str, notes: str, user_input: str = "") -> str:
    prompt = build_llm_prompt(label=label, notes=notes, user_input=user_input)
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("LLM response was empty.")
    return content.strip()


def load_notes_from_metadata(video_path: Path) -> str:
    metadata_path = video_path.with_suffix(".json")
    if not metadata_path.exists():
        return ""

    with metadata_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    notes_parts = [
        data.get("primary_issue"),
        data.get("primary_fix"),
        data.get("primary_consequence"),
        data.get("secondary_issue"),
        data.get("secondary_fix"),
        data.get("secondary_note"),
    ]
    notes = "; ".join([part for part in notes_parts if part])
    return notes
