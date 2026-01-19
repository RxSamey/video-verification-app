import os
import re
import tempfile
import base64
import hashlib
from io import BytesIO
from typing import List, Dict

import streamlit as st
from openai import OpenAI
import pdfplumber
import cv2
from PIL import Image


# ===================== SESSION STATE =====================
if "running" not in st.session_state:
    st.session_state.running = False

if "audit_started" not in st.session_state:
    st.session_state.audit_started = False


# ===================== HELPERS =====================

def extract_test_case_id(ticket_text: str) -> str:
    if not ticket_text:
        return "UNKNOWN"
    first_line = ticket_text.strip().splitlines()[0]
    first_token = first_line.strip().split()[0]
    if re.match(r"[A-Z]{2,20}-[A-Z]*\d+", first_token):
        return first_token
    match = re.search(r"\b[A-Z]{2,20}-[A-Z]*\d+\b", ticket_text)
    return match.group(0) if match else "UNKNOWN"


def extract_test_steps(ticket_text: str) -> List[str]:
    steps: List[str] = []
    for line in ticket_text.splitlines():
        if re.match(r"^\s*(\d+\.|-|\*)\s+", line):
            step = re.sub(r"^\s*(\d+\.|-|\*)\s+", "", line).strip()
            if len(step) > 4:
                steps.append(step)
    return steps


def classify_step(step: str) -> str:
    s = step.lower()

    backend_keywords = [
        "backend", "database", "server", "api", "log",
        "timestamp", "earnings", "ufp", "should match", "matches with"
    ]
    implicit_keywords = [
        "login", "log in", "credential", "mock location",
        "authenticated", "accepted"
    ]

    if any(k in s for k in backend_keywords):
        return "backend"
    if any(k in s for k in implicit_keywords):
        return "implicit"
    return "visual"


def calculate_step_coverage(step_results: List[Dict]) -> int:
    scorable = [r for r in step_results if r["type"] in ("visual", "implicit")]
    if not scorable:
        return 0
    covered = sum(1 for r in scorable if r["status"] == "Covered")
    return int((covered / len(scorable)) * 100)


def derive_verdict(step_results: List[Dict]) -> str:
    scorable = [r for r in step_results if r["type"] in ("visual", "implicit")]
    if not scorable:
        return "No Match"
    if all(r["status"] == "Covered" for r in scorable):
        return "Match"
    if any(r["status"] == "Covered" for r in scorable):
        return "Partial Match"
    return "No Match"


def calculate_confidence(step_results: List[Dict]) -> str:
    visual = [r for r in step_results if r["type"] == "visual"]
    implicit = [r for r in step_results if r["type"] == "implicit"]

    if not visual:
        return "Low"

    covered_visual = sum(1 for r in visual if r["status"] == "Covered")
    ratio = covered_visual / len(visual)

    if ratio >= 0.85 and len(implicit) <= 1:
        return "High"
    if ratio >= 0.6:
        return "Medium"
    return "Low"


def format_compact_result(test_case_id, verdict, coverage, confidence, step_results):
    passed = [r for r in step_results if r["status"] == "Covered" and r["type"] != "backend"]
    failed = [r for r in step_results if r["status"] == "Not Covered" and r["type"] != "backend"]
    backend = [r for r in step_results if r["type"] == "backend"]

    html = f"""
<b>Test Case ID:</b> {test_case_id}<br>
<b>Status:</b> {verdict}<br>
<b>Coverage:</b> {coverage}%<br>
<b>Confidence:</b> {confidence}<br>
"""

    if passed:
        html += "<b>Passed Steps:</b><ul>"
        for r in passed:
            html += f"<li>✔ {r['step']}</li>"
        html += "</ul>"

    if failed:
        html += "<b>Failed Steps:</b><ul>"
        for r in failed:
            html += f"<li>✖ {r['step']}</li>"
        html += "</ul>"

    if backend:
        html += "<b>Backend / Manual Validation:</b><ul>"
        for r in backend:
            html += f"<li>• {r['step']}</li>"
        html += "</ul>"

    return html


def read_ticket_file(path: str, ext: str) -> str:
    if ext.lower() == "pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        return text
    return open(path, encoding="utf-8", errors="ignore").read()


def extract_frames(video_path: str) -> List[Image.Image]:
    frames: List[Image.Image] = []
    cap = cv2.VideoCapture(video_path)

    success, frame = cap.read()
    prev = None

    while success:
        if prev is None or cv2.absdiff(frame, prev).mean() > 5:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            prev = frame
        success, frame = cap.read()

    cap.release()
    return frames


@st.cache_data(show_spinner=False)
def cached_extract_frames(video_path: str) -> List[Image.Image]:
    return extract_frames(video_path)


def pil_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze_video_once(client: OpenAI, frames: List[Image.Image]) -> str:
    sampled = frames[::10][:3] if len(frames) > 10 else frames[:3]

    prompt = """
You are a QA auditor.

Analyze the screenshots from a mobile app screen recording.
Describe:
- Visible screens
- Navigation flow
- Key UI states (service tab, tiles, pick-up/drop-off screens, dialogs)
- What is visible and what is NOT visible
"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            *[
                {"type": "image_url", "image_url": {"url": pil_to_base64(f)}}
                for f in sampled
            ]
        ]
    }]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,  # type: ignore
        temperature=0,
        timeout=30
    )

    return response.choices[0].message.content.strip()


def get_video_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@st.cache_data(show_spinner=False)
def cached_video_summary(video_hash: str, frames: List[Image.Image], api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    return analyze_video_once(client, frames)


def verify_steps_from_summary(client: OpenAI, steps: List[str], summary: str) -> List[Dict]:
    results: List[Dict] = []

    for step in steps:
        step_type = classify_step(step)

        if step_type == "backend":
            results.append({"step": step, "type": "backend", "status": "Not Scored"})
            continue

        if step_type == "implicit":
            results.append({"step": step, "type": "implicit", "status": "Covered"})
            continue

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{step}\n\nSUMMARY:\n{summary}\n\nCovered or Not Covered?"}],
            temperature=0,
            timeout=15
        )

        status = response.choices[0].message.content.strip()
        if status not in ("Covered", "Not Covered"):
            status = "Not Covered"

        results.append({"step": step, "type": step_type, "status": status})

    return results


# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Test Script vs Video Verification", layout="wide")
st.title("🎥 Test Script Verification Using Video Artefact")

api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not api_key:
    st.stop()

client = OpenAI(api_key=api_key)

st.warning("⚠️ Vision analysis uses OpenAI credits (~$2–$5 per run).")
confirm_run = st.checkbox("I understand the cost and want to proceed")

st.header("1️⃣ Test Case Input")
mode = st.radio("Input method", ["Paste Ticket Text", "Upload Ticket Files"])

tickets: List[str] = []
if mode == "Paste Ticket Text":
    pasted = st.text_area("Paste test case", height=260)
    if pasted.strip():
        tickets = [pasted.strip()]
else:
    files = st.file_uploader("Upload tickets", type=["txt", "pdf"], accept_multiple_files=True)
    for f in files or []:
        ext = f.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+ext) as tmp:
            tmp.write(f.read())
            tickets.append(read_ticket_file(tmp.name, ext))

st.header("2️⃣ Video Artefacts")
videos = st.file_uploader("Upload videos", type=["mp4", "mov"], accept_multiple_files=True)
video_paths: List[str] = []
for v in videos or []:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(v.read())
        video_paths.append(tmp.name)

st.header("3️⃣ Run Verification")
run = st.button(
    "▶ Run Verification",
    disabled=st.session_state.running or st.session_state.audit_started
)

if run:
    if not confirm_run:
        st.error("Please confirm cost acknowledgement.")
        st.stop()

    st.session_state.running = True
    st.session_state.audit_started = True

if st.session_state.running:
    with st.spinner("Processing…"):

        for i, ticket in enumerate(tickets):
            video = video_paths[i] if i < len(video_paths) else video_paths[-1]

            frames = cached_extract_frames(video)
            video_hash = get_video_hash(video)
            summary = cached_video_summary(video_hash, frames, api_key)

            steps = extract_test_steps(ticket)
            step_results = verify_steps_from_summary(client, steps, summary)

            verdict = derive_verdict(step_results)
            coverage = calculate_step_coverage(step_results)
            confidence = calculate_confidence(step_results)

            st.markdown(format_compact_result(
                extract_test_case_id(ticket),
                verdict,
                coverage,
                confidence,
                step_results
            ), unsafe_allow_html=True)

    st.session_state.running = False
    st.session_state.audit_started = False
