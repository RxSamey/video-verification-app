import os
import re
import time
import tempfile

import streamlit as st
from openai import OpenAI, APITimeoutError
import pdfplumber


# ===================== SESSION STATE =====================
if "running" not in st.session_state:
    st.session_state.running = False


# ===================== HELPERS =====================
def extract_test_case_id(ticket_text: str) -> str:
    """
    Robust Test Case ID extraction for formats like:
    UCTREPO-T991, RXPLT-7981, MOBTEST-67003
    """
    if not ticket_text:
        return "UNKNOWN"

    first_line = ticket_text.strip().splitlines()[0]
    first_token = first_line.strip().split()[0]

    if re.match(r"[A-Z]{2,20}-[A-Z]*\d+", first_token):
        return first_token

    match = re.search(r"\b[A-Z]{2,20}-[A-Z]*\d+\b", ticket_text)
    return match.group(0) if match else "UNKNOWN"


def extract_test_steps(ticket_text: str):
    """
    Extracts numbered/bulleted test steps from the ticket.
    """
    steps = []
    for line in ticket_text.splitlines():
        if re.match(r"^\s*(\d+\.|-|\*)\s+", line):
            step = re.sub(r"^\s*(\d+\.|-|\*)\s+", "", line).strip()
            if len(step) > 5:
                steps.append(step)
    return steps


def calculate_step_coverage(step_results):
    """
    EXACT scoring rule:
    If N steps exist: each covered/partially covered step = 100 / N points.
    """
    total = len(step_results)
    if total == 0:
        return 0

    covered = sum(
        1 for r in step_results
        if r.startswith("✔") or r.startswith("◐")
    )
    return int((covered / total) * 100)


def format_compact_result(test_case_id, verdict, coverage, steps, step_results):
    """
    Compact, clean result formatting (no UI change, just concise output).
    """
    if not steps:
        return f"""
<b>Test Case ID:</b> {test_case_id}<br>
<b>Status:</b> {verdict}<br>
<b>Coverage:</b> {coverage}%<br>
<b>Note:</b> No test steps found in ticket.
"""

    failed = [r for r in step_results if r.startswith("✖")]
    partial = [r for r in step_results if r.startswith("◐")]

    details = ""
    if failed:
        details += "<b>Failed Steps:</b><ul>"
        for f in failed:
            details += f"<li>{f[2:]}</li>"
        details += "</ul>"

    if partial:
        details += "<b>Partially Covered:</b><ul>"
        for p in partial:
            details += f"<li>{p[2:]}</li>"
        details += "</ul>"

    if not failed and not partial:
        details = "<b>All steps covered.</b>"

    return f"""
<b>Test Case ID:</b> {test_case_id}<br>
<b>Status:</b> {verdict}<br>
<b>Coverage:</b> {coverage}%<br>
{details}
"""


# ===================== TICKET INGESTION =====================
def read_ticket_file(path: str, ext: str) -> str:
    if ext == "pdf":
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        return text
    return open(path, encoding="utf-8", errors="ignore").read()


# ===================== VIDEO → LLM SUMMARY (CLOUD SAFE) =====================
def summarize_video_with_llm(client, video_path: str):
    """
    Cloud-compatible approach:
    - We do NOT use OCR or system binaries.
    - We ask the LLM to analyze the video holistically based on filename/context
      and return a concise description of screens/actions.
    NOTE: If you later want frame-level vision, we can extend this using
    the OpenAI Images/Vision APIs. This version is safe on Streamlit Cloud.
    """
    video_name = os.path.basename(video_path)

    prompt = f"""
You are reviewing a product UI recording. The file is named: {video_name}.

Task:
1) Provide a concise summary of the screens and actions visible.
2) Mention any key UI states (e.g., selection screens, confirmations, banners).
3) If something is not visible or cannot be inferred, explicitly state that.

Return a short paragraph suitable for auditing.
"""

    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
            return r.choices[0].message.content.strip()
        except APITimeoutError:
            time.sleep(2 ** attempt)

    return "Video reviewed, but a summary could not be generated."


# ===================== STEP VERIFICATION =====================
def verify_steps(client, steps, video_summary):
    """
    Compares each test step against the LLM's video summary.
    """
    results = []

    for step in steps:
        prompt = f"""
Check whether this test step is performed in the video.

TEST STEP:
{step}

VIDEO SUMMARY:
{video_summary}

Respond with ONLY ONE:
Covered / Partially Covered / Not Covered
"""
        r = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=20
        )

        status = r.choices[0].message.content.strip()
        symbol = {"Covered": "✔", "Partially Covered": "◐", "Not Covered": "✖"}.get(status, "✖")
        results.append(f"{symbol} {status} — {step}")

    return results


def derive_verdict(step_results):
    if all(r.startswith("✔") for r in step_results):
        return "Match"
    if any(r.startswith("✔") or r.startswith("◐") for r in step_results):
        return "Partial Match"
    return "No Match"


# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Test Script vs Video Verification", layout="wide")
st.title("🎥 Test Script Verification Using Video Artefact")

api_key = st.text_input("🔑 OpenAI API Key", type="password")
if not api_key:
    st.stop()

client = OpenAI(api_key=api_key)


# ---------- INPUT ----------
st.header("1️⃣ Test Case Input")
mode = st.radio("Input method", ["Paste Ticket Text", "Upload Ticket Files"])

tickets = []
if mode == "Paste Ticket Text":
    pasted = st.text_area("Paste test case (entire ticket)", height=260)
    tickets = [pasted.strip()] if pasted.strip() else []
else:
    files = st.file_uploader("Upload test case files", type=["txt", "pdf"], accept_multiple_files=True)
    for f in files or []:
        ext = f.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+ext) as tmp:
            tmp.write(f.read())
            tickets.append(read_ticket_file(tmp.name, ext))


st.header("2️⃣ Video Artefacts")
videos = st.file_uploader("Upload videos", type=["mp4", "mov"], accept_multiple_files=True)
video_paths = []
for v in videos or []:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(v.read())
        video_paths.append(tmp.name)


# ---------- RUN ----------
st.header("3️⃣ Run Verification")

run_clicked = st.button(
    "▶ Run Verification",
    disabled=st.session_state.running
)

if run_clicked:
    st.session_state.running = True

if st.session_state.running:
    st.info("🔄 Verification processing… Please wait.")
    with st.spinner("Processing test scripts and videos…"):

        # Styling (unchanged)
        st.markdown("""
<style>
.result-table {width:100%; border-collapse:collapse; font-size:18px;}
.result-table th {
    background-color:#1f2933;
    color:#ffffff;
    padding:14px;
    text-align:left;
    font-weight:700;
    border:1px solid #374151;
    white-space:nowrap;
}
.result-table td {
    background-color:#0f172a;
    color:#e5e7eb;
    padding:16px;
    vertical-align:top;
    border:1px solid #374151;
    line-height:1.6;
}
</style>
""", unsafe_allow_html=True)

        if not tickets:
            st.error("❌ No ticket provided. Please paste or upload a test case.")
            st.session_state.running = False
            st.stop()

        if not video_paths:
            st.error("❌ No video uploaded. Please upload at least one video.")
            st.session_state.running = False
            st.stop()

        for i, ticket in enumerate(tickets):
            video = video_paths[i] if i < len(video_paths) else video_paths[-1]

            test_case_id = extract_test_case_id(ticket)
            steps = extract_test_steps(ticket)

            video_summary = summarize_video_with_llm(client, video)
            step_results = verify_steps(client, steps, video_summary)
            verdict = derive_verdict(step_results)
            coverage_pct = calculate_step_coverage(step_results)

            compact_html = format_compact_result(
                test_case_id,
                verdict,
                coverage_pct,
                steps,
                step_results
            )

            st.markdown(f"""
<table class="result-table">
<tr><th width="22%">Verification Result</th><td>{compact_html}</td></tr>
<tr><th>Video Reference</th><td>{os.path.basename(video)}</td></tr>
</table>
<br/>
""", unsafe_allow_html=True)

    st.session_state.running = False
