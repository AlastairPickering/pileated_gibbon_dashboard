import streamlit as st
from pathlib import Path

st.set_page_config(page_title="PAMalytics Studio", layout="wide")

# Simple global config for Studio (kept under studio/)
STUDIO_ROOT = Path(__file__).resolve().parent
PROJECTS_ROOT = STUDIO_ROOT / "projects"
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

# Light top bar
st.markdown(
    """
    <style>
      .topbar {display:flex;gap:1rem;align-items:center; margin:-1rem 0 1rem 0;}
      .badge {padding:2px 8px;border-radius:999px;background:#eef;border:1px solid #ccd;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="topbar"><span class="badge">Studio</span><b>PAMalytics</b></div>', unsafe_allow_html=True)

st.title("PAMalytics Studio")
st.write("Login → Project Hub → Workspace. Data import comes next.")
st.info("Use the sidebar to navigate. Start at **🔐 Login** then **🏠 Project Hub**.")
