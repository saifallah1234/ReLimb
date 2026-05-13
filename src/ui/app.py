from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import settings


st.set_page_config(page_title="ReLimb LLM Summary", page_icon="🦿")

st.title("ReLimb Gait Summary")
st.write("Upload a walking video to get the LLM-generated summary.")

api_url = st.text_input("API URL", value=f"http://{settings.api_host}:{settings.api_port}")
user_input = st.text_area(
    "Optional: your message (used to detect response language)",
    placeholder="Example: مرحبا، أريد شرح النتيجة",
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file and st.button("Analyze video"):
    with st.spinner("Uploading and analyzing... this can take a few minutes"):
        files = {
            "video": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "video/mp4"),
        }
        data: dict[str, Any] = {"user_input": user_input}
        response = requests.post(f"{api_url}/predict", files=files, data=data, timeout=900)

    if response.ok:
        payload = response.json()
        st.success("Analysis complete")
        st.subheader("Detected label")
        st.write(payload.get("label", "Unknown"))
        st.subheader("LLM Summary")
        st.write(payload.get("summary", ""))
    else:
        st.error(f"Request failed: {response.status_code} - {response.text}")
