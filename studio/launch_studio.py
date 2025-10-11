# studio/launch_studio.py
# Flow: Login → Project Hub → Overview → Data mapping → Audio mapping → Metadata mapping → Dashboard
# Studio (import steps) has no sidebar/header; PAMalytics pages restore them. Python 3.9-compatible. UK English.

import os
import sys
import json
import uuid
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, NamedTuple

# ──────────────────────────────────────────────────────────────────────────────
# Dual-mode launcher:
# - If PA_STUDIO_AS_APP != "1": pure Python launcher (no Streamlit import/UI), does
#   venv bootstrap + installs + execs Streamlit ONCE. => No flash.
# - If PA_STUDIO_AS_APP == "1": we are inside Streamlit app; run Studio UI.
# ──────────────────────────────────────────────────────────────────────────────

STUDIO_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = STUDIO_ROOT.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

def _default_venv_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "PileatedGibbonDashboard" / ".venv"
    return Path.home() / ".pileated_gibbon_dashboard" / ".venv"

def _bootstrap_and_exec_streamlit():
    import venv as _venv
    from pathlib import Path as _Path
    import subprocess as _sp

    _STUDIO_FILE = _Path(__file__).resolve()
    _REPO_ROOT   = _STUDIO_FILE.parent.parent
    _SCRIPTS_DIR = _REPO_ROOT / "scripts"

    _REQS_FILE = _SCRIPTS_DIR / "requirements.txt"
    if not _REQS_FILE.exists() and (_REPO_ROOT / "requirements.txt").exists():
        _REQS_FILE = _REPO_ROOT / "requirements.txt"

    _VENV_DIR = _Path(os.environ.get("PG_VENV_DIR", str(_default_venv_dir())))
    _PY_EXE   = _VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _run(cmd, **kw):
        print(">", " ".join(map(str, cmd)))
        _sp.check_call(cmd, **kw)

    # create venv if needed
    if not _VENV_DIR.exists():
        print(f"[setup] Creating virtual environment at: {_VENV_DIR}")
        _VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        _venv.EnvBuilder(with_pip=True).create(str(_VENV_DIR))

    # upgrade pip tooling
    _run([str(_PY_EXE), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # install requirements
    if not _REQS_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {_REQS_FILE}")
    print("[setup] Installing requirements.txt …")
    _run([str(_PY_EXE), "-m", "pip", "install", "-r", str(_REQS_FILE)])

    # optional: torch check (safe to fail)
    def _torch_import_ok(py_exe: str) -> bool:
        try:
            _run([py_exe, "-c", "import torch; print(torch.__version__)"])
            return True
        except _sp.CalledProcessError:
            return False

    def _install_torch(py_exe: str):
        print("[setup] Installing PyTorch…")
        if platform.system() == "Windows":
            _run([py_exe, "-m", "pip", "install",
                  "--index-url", "https://download.pytorch.org/whl/cpu",
                  "torch", "torchvision", "torchaudio"])
        else:
            _run([py_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        _run([py_exe, "-c", "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__)"])

    try:
        if not _torch_import_ok(str(_PY_EXE)):
            _install_torch(str(_PY_EXE))
    except Exception as _e:
        print(f"[setup] PyTorch check/install skipped or failed: {_e}")

    # exec Streamlit ONCE (no flash)
    env = os.environ.copy()
    env["PA_STUDIO_AS_APP"] = "1"   # signal: now run the Streamlit app branch
    env["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
    env["STREAMLIT_LOG_LEVEL"] = "error"
    PORT = str(env.get("PA_STUDIO_PORT", "8510"))
    args = [
        str(_PY_EXE), "-m", "streamlit", "run", str(_STUDIO_FILE),
        "--server.headless", "true",
        "--server.port", PORT,
        "--server.fileWatcherType", "none",
        "--logger.level", "error",
    ]
    os.execvpe(str(_PY_EXE), args, env)  # replace current process forever

# If we are NOT inside the Streamlit app, do bootstrap + start Streamlit once.
if os.environ.get("PA_STUDIO_AS_APP") != "1":
    _bootstrap_and_exec_streamlit()
    raise SystemExit

# ──────────────────────────────────────────────────────────────────────────────
# From here, we are inside Streamlit (env PA_STUDIO_AS_APP=1). App code below.
# ──────────────────────────────────────────────────────────────────────────────

# Make scripts/ importable
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import streamlit as st

# Studio: hide chrome; PAMalytics pages re-enable it explicitly.
st.set_page_config(page_title="PAMalytics Studio", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      header, footer, [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display:none !important; }
      .block-container { padding-left: 1rem; padding-right: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Small helpers ----------
def chip(text: str, kind: str = "info") -> str:
    colours = {"ready":"#16a34a","pending":"#d97706","empty":"#6b7280","error":"#dc2626","info":"#3b82f6"}
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{colours.get(kind, "#3b82f6")};color:white;font-size:12px">{text}</span>'

def _btn(label: str, key: Optional[str] = None) -> bool:
    return st.button(label, key=key or label)

def nav_row(left_label: str, left_route: str, right_label: Optional[str] = None, right_route: Optional[str] = None, key_prefix: str = "nav"):
    c1, c2 = st.columns([1, 1])
    if left_label and c1.button(left_label, key=f"{key_prefix}_left"):
        st.session_state.route = left_route
        st.rerun()
    if right_label and c2.button(right_label, key=f"{key_prefix}_right"):
        st.session_state.route = right_route
        st.rerun()

# ---------- Projects / storage ----------
PROJECTS_ROOT = STUDIO_ROOT / "projects"
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
AUTH_FILE = STUDIO_ROOT / ".auth.json"

@dataclass
class ProjectManifest:
    project_id: str
    name: str
    use_case: str
    tz: str = "UTC"
    created_by: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())
    last_opened: Optional[str] = None
    paths: Optional[dict] = None
    status: Optional[dict] = None
    provenance: Optional[dict] = None
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

def _slug(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name.strip())
    return s[:64] or "project"

def _default_status() -> Dict[str, str]:
    return {"import_results":"empty","audio_resolver":"empty","metadata_joins":"empty","analysis":"empty","export":"empty"}

def _default_paths(folder: Path) -> Dict[str, str]:
    return {"root": str(folder), "data_raw":"data_raw/","data_normalised":"data_normalised/","metadata":"metadata/","exports":"exports/","logs":"logs/","workspace":"workspace/"}

def create_project(name: str, use_case: str, created_by: Optional[str]) -> Path:
    folder = PROJECTS_ROOT / _slug(name); folder.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest(project_id=str(uuid.uuid4()), name=name, use_case=use_case, created_by=created_by or "user",
                               paths=_default_paths(folder), status=_default_status(),
                               provenance={"app":"pamalytics_studio","version":"0.3.0"},
                               last_opened=datetime.now(dt_timezone.utc).isoformat())
    for p in manifest.paths.values(): (Path(folder) / p).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "project.json").write_text(manifest.to_json(), encoding="utf-8"); return Path(folder)

def list_projects() -> List[Path]:
    return sorted([p for p in PROJECTS_ROOT.iterdir() if (p / "project.json").exists()], key=lambda p: p.stat().st_mtime, reverse=True)

def load_project(folder: Path) -> dict:
    return json.loads((folder / "project.json").read_text(encoding="utf-8"))

def save_project(folder: Path, data: Dict[str, Any]) -> None:
    (folder / "project.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

def touch_last_opened(folder: Path) -> None:
    data = load_project(folder); data["last_opened"] = datetime.now(dt_timezone.utc).isoformat(); save_project(folder, data)

def set_status(folder: Path, key: str, value: str) -> None:
    data = load_project(folder)
    if "status" not in data or not isinstance(data["status"], dict): data["status"] = _default_status()
    data["status"][key] = value; save_project(folder, data)

def ensure_paths_schema(folder: Path) -> None:
    data = load_project(folder); paths = data.get("paths") or {}; changed = False
    for k, v in _default_paths(folder).items():
        if k not in paths: paths[k] = v; changed = True
    if changed:
        data["paths"] = paths; save_project(folder, data)
        for rel in paths.values(): (folder / rel).mkdir(parents=True, exist_ok=True)

def project_path(folder: Path, *keys: str) -> Path:
    ensure_paths_schema(folder); data = load_project(folder)
    base = (folder / data["paths"][keys[0]]).resolve(); base.mkdir(parents=True, exist_ok=True)
    for k in keys[1:]: base = (base / k).resolve()
    return base

# ---------- Session boot ----------
ss = st.session_state
ss.setdefault("auth_user", None)
ss.setdefault("route", "login")
ss.setdefault("current_project", None)
ss.setdefault("import_params", {})
ss.setdefault("import_preview_ready", False)
ss.setdefault("import_preview_df", None)
ss.setdefault("import_last_saved", None)
ss.setdefault("import_notes", [])
ss.setdefault("audio_dirs", [])
ss.setdefault("audio_map_df", None)
ss.setdefault("audio_map_preview", None)
ss.setdefault("audio_save_path", None)
ss.setdefault("use_stem_fallback", True)

# Remember-me
AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
if ss.get("auth_user") is None and AUTH_FILE.exists():
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        if data.get("remember") and data.get("user"):
            ss.auth_user = data["user"]
            ss.route = "hub"
    except Exception:
        pass

# ---------- Helpers for matching ----------
class Coverage(NamedTuple):
    matched_rows: int
    total_rows: int
    matched_unique_files: int
    total_unique_files: int

def analysis_keys(df, col="source_file"):
    import pandas as pd, os
    out = df.copy()
    out["_basename"] = out[col].astype(str).apply(lambda p: os.path.basename(p).strip())
    out["_name_lower"] = out["_basename"].str.lower()
    out["_stem_lower"] = out["_name_lower"].apply(lambda s: os.path.splitext(s)[0])
    return out

def compute_audio_coverage(detections_csv: Path, mapping: Any, use_stem_fallback: bool = True) -> Coverage:
    import pandas as pd
    det = pd.read_csv(detections_csv)
    if det.empty or "source_file" not in det.columns: return Coverage(0,0,0,0)
    det = analysis_keys(det)
    total_rows = int(len(det))
    det_files = det[["_basename","_name_lower","_stem_lower"]].drop_duplicates()
    total_unique_files = int(len(det_files))

    if hasattr(mapping, "to_dict"): mp = mapping.copy()
    elif isinstance(mapping, (str, Path)):
        try: mp = pd.read_csv(mapping)
        except Exception: return Coverage(0, total_rows, 0, total_unique_files)
    else: return Coverage(0, total_rows, 0, total_unique_files)

    if mp.empty or "filename" not in mp.columns: return Coverage(0, total_rows, 0, total_unique_files)
    mp = mp.assign(_filename=mp["filename"].astype(str).str.strip())
    mp["_name_lower"] = mp["_filename"].str.lower()
    mp["_stem_lower"] = mp["_name_lower"].apply(lambda s: os.path.splitext(s)[0])

    name_set = set(mp["_name_lower"].unique())
    name_match_mask = det["_name_lower"].isin(name_set)

    if use_stem_fallback:
        stem_counts = mp["_stem_lower"].value_counts()
        unique_stems = set(stem_counts[stem_counts == 1].index)
        stem_match_mask = det["_stem_lower"].isin(unique_stems)
        files_stem_match = det_files["_stem_lower"].isin(unique_stems)
    else:
        stem_match_mask = det["_stem_lower"].isin(set())
        files_stem_match = False

    final_match_mask = name_match_mask | (~name_match_mask & stem_match_mask)
    matched_rows = int(final_match_mask.sum())
    files_name_match = det_files["_name_lower"].isin(name_set)
    files_final = files_name_match | (~files_name_match & files_stem_match)
    matched_unique_files = int(files_final.sum())
    return Coverage(matched_rows, total_rows, matched_unique_files, total_unique_files)

def compute_import_stats(norm_csv: Path, audio_csv: Optional[Path], meta_csv: Optional[Path], use_stem_fallback: bool=True) -> Dict[str, Any]:
    import pandas as pd
    stats = {"detections_rows":0,"unique_files_in_detections":0,"audio_files_indexed":0,"detections_with_audio":0,"metadata_join_rows":0,"final_rows":0}
    if not norm_csv.exists(): return stats

    det = pd.read_csv(norm_csv)
    if det.empty or "source_file" not in det.columns: return stats
    det = analysis_keys(det)
    stats["detections_rows"] = len(det)
    stats["unique_files_in_detections"] = det["_basename"].nunique()

    if audio_csv and Path(audio_csv).exists():
        mp = pd.read_csv(audio_csv)
        if not mp.empty and "filename" in mp.columns:
            stats["audio_files_indexed"] = int(mp["filename"].nunique())
            mp = mp.assign(_filename=mp["filename"].astype(str).str.strip().str.lower())
            mp["_stem_lower"] = mp["_filename"].apply(lambda s: os.path.splitext(s)[0])
            name_set = set(mp["_filename"].unique())
            mask = det["_name_lower"].isin(name_set)
            if use_stem_fallback:
                stem_counts = mp["_stem_lower"].value_counts()
                unique_stems = set(stem_counts[stem_counts == 1].index)
                mask = mask | (~mask & det["_stem_lower"].isin(unique_stems))
            with_audio = det.loc[mask].copy()
            stats["detections_with_audio"] = len(with_audio)

            if meta_csv and Path(meta_csv).exists():
                meta = pd.read_csv(meta_csv)
                stats["metadata_join_rows"] = len(meta)

                with_audio = with_audio.copy()
                with_audio["_det_id"] = with_audio.index
                with_audio["basename"] = with_audio["_basename"]
                with_audio["stem"] = with_audio["_stem_lower"]
                with_audio["recorder_id"] = with_audio["basename"].apply(lambda n: n.split("_", 1)[0] if "_" in n else n)

                join_key_det, join_key_meta = None, None
                if "recorder_id" in meta.columns:
                    join_key_det, join_key_meta = "recorder_id", "recorder_id"
                elif "basename" in meta.columns:
                    join_key_det, join_key_meta = "basename", "basename"
                elif "filename" in meta.columns:
                    join_key_det, join_key_meta = "basename", "filename"
                elif "stem" in meta.columns:
                    join_key_det, join_key_meta = "stem", "stem"

                if join_key_det and join_key_meta:
                    meta_keys = meta[[join_key_meta]].dropna().astype(str).drop_duplicates()
                    joined = with_audio.merge(meta_keys, left_on=join_key_det, right_on=join_key_meta, how="left", indicator=True)
                    matched_det_count = int(joined.loc[joined["_merge"] == "both", "_det_id"].nunique())
                    stats["final_rows"] = matched_det_count
                else:
                    stats["final_rows"] = len(with_audio)
            else:
                stats["final_rows"] = len(with_audio)
    return stats

def build_analysis_dataset(proj_path: Path, use_stem_fallback: bool=True):
    """
    Return (df, notes) filtered to rows that have a mapped audio file.
    If enriched detections exist, use those; else use normalised.
    """
    import pandas as pd
    norm = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    enriched = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
    audio_csv = project_path(proj_path, "workspace") / "audio_paths.csv"

    if not norm.exists() and not enriched.exists():
        return None, ["No detections found."]
    if not audio_csv.exists():
        return None, ["No audio mapping found."]

    det = pd.read_csv(enriched if enriched.exists() else norm)
    mp = pd.read_csv(audio_csv)
    if det.empty:
        return None, ["Detections are empty."]
    if mp.empty or "filename" not in mp.columns:
        return None, ["Audio mapping is empty or missing the 'filename' column."]

    notes = []
    det = analysis_keys(det)
    mp = mp.assign(_filename=mp["filename"].astype(str).str.strip().str.lower())
    mp["_stem_lower"] = mp["_filename"].apply(lambda s: os.path.splitext(s)[0])

    name_set = set(mp["_filename"].unique())
    mask = det["_name_lower"].isin(name_set)
    if use_stem_fallback:
        stem_counts = mp["_stem_lower"].value_counts()
        unique_stems = set(stem_counts[stem_counts == 1].index)
        stem_mask = det["_stem_lower"].isin(unique_stems)
        mask = mask | (~mask & stem_mask)
        if (det["_stem_lower"].isin(set(stem_counts.index) - unique_stems)).any():
            notes.append("Some stems were ambiguous and therefore not auto-matched.")

    matched = det.loc[mask].copy()

    left = matched.merge(mp[["_filename","path"]], left_on="_name_lower", right_on="_filename", how="left")
    if use_stem_fallback:
        need = left["path"].isna()
        if need.any():
            stem_join = left.loc[need, ["_stem_lower"]].merge(mp[["_stem_lower", "path"]], on="_stem_lower", how="left")
            left.loc[need, "path"] = stem_join["path"].values

    left["basename"]   = left["_basename"]
    left["stem"]       = left["_stem_lower"]
    left["recorder_id"] = left["basename"].apply(lambda n: n.split("_", 1)[0] if "_" in n else n)
    return left, notes

# ------------------ Views ------------------
def view_login() -> None:
    st.title("Login")
    default_user = ""
    if AUTH_FILE.exists():
        try:
            prev = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
            default_user = prev.get("user", "")
        except Exception:
            pass
    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Username", value=default_user)
        st.text_input("PIN (optional)", type="password")  # retained but unused
        remember = st.checkbox("Remember me", value=True)
        submit = st.form_submit_button("Sign in")
    if submit:
        if not user.strip():
            st.error("Please enter a username.")
        else:
            st.session_state.auth_user = user.strip()
            try:
                AUTH_FILE.write_text(json.dumps({"remember": bool(remember), "user": st.session_state.auth_user}), encoding="utf-8")
            except Exception:
                pass
            st.session_state.route = "hub"
            st.rerun()

def view_hub() -> None:
    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"; st.rerun()
    st.title("Project Hub")
    with st.expander("Create a new project", expanded=True):
        name = st.text_input("Project name", placeholder="e.g. Sabah 2024 – external results", key="proj_name")
        mode = st.radio(
            "What would you like to do?",
            options=["external_results", "pipeline"],
            index=0,
            format_func=lambda x: "I already have classifier results" if x == "external_results" else "I have audio; I want to run a classifier",
            horizontal=True,
        )
        if _btn("Create project", key="create_project_btn") and name.strip():
            folder = create_project(name.strip(), mode, created_by=st.session_state.auth_user)
            touch_last_opened(folder)
            st.session_state.current_project = str(folder)
            st.session_state.route = "overview"
            st.success(f"Created project: `{folder.name}`")
            st.rerun()

    st.subheader("Recent projects")
    projects = list_projects()
    if not projects:
        st.caption("No projects yet. Create one above.")
    else:
        for p in projects:
            data = load_project(p)
            cols = st.columns([3, 2, 2, 1])
            cols[0].markdown(f"**{data.get('name','(unnamed)')}**  \n`{p.name}`")
            cols[1].write(f"Mode: `{data.get('use_case')}`")
            cols[2].write(f"Timezone: `{data.get('tz','UTC')}`")
            if cols[3].button("Open", key=f"open_{p.name}"):
                st.session_state.current_project = str(p)
                touch_last_opened(p)
                st.session_state.route = "overview"
                st.toast(f"Opened: {p.name}")
                st.rerun()

    if st.session_state.get("current_project"):
        st.success(f"Active project: `{Path(st.session_state.current_project).name}`")

def _import_progress_cards(proj_path: Path) -> Dict[str, str]:
    data = load_project(proj_path)
    s = (data.get("status") or _default_status()).copy()
    norm_csv    = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    audio_csv   = project_path(proj_path, "workspace")       / "audio_paths.csv"
    enriched    = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
    if norm_csv.exists(): s["import_results"]  = "ready"
    if audio_csv.exists(): s["audio_resolver"] = "ready"
    if enriched.exists():  s["metadata_joins"] = "ready"
    return s

def view_overview() -> None:
    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"; st.rerun()
    if not st.session_state.get("current_project"):
        st.session_state.route = "hub"; st.rerun()

    proj_path = Path(st.session_state.current_project)
    ensure_paths_schema(proj_path)
    data = load_project(proj_path)

    st.title("Overview")
    st.caption(f"Project: **{data['name']}** • Mode: `{data['use_case']}` • Timezone: `{data.get('tz','UTC')}`")

    st.write("### Import phase (3 steps)")
    st.caption("Complete these in order: **1) Data mapping** → **2) Audio mapping** → **3) Metadata mapping**.")
    s = _import_progress_cards(proj_path)

    norm_csv  = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    audio_csv = project_path(proj_path, "workspace")       / "audio_paths.csv"
    enrich_csv= project_path(proj_path, "data_normalised") / "detections_enriched.csv"

    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("#### 1) Data mapping")
        st.markdown(chip("ready" if s["import_results"]=="ready" else ("pending" if s["import_results"]=="pending" else "empty"),
                         "ready" if s["import_results"]=="ready" else ("pending" if s["import_results"]=="pending" else "empty")),
                    unsafe_allow_html=True)
        if norm_csv.exists(): st.caption(f"`{norm_csv.name}`")
        if st.button("Open Data mapping", key="open_step1"):
            st.session_state.route = "import"; st.rerun()

    with step2:
        st.markdown("#### 2) Audio mapping")
        st.markdown(chip("ready" if s["audio_resolver"]=="ready" else ("pending" if s["audio_resolver"]=="pending" else "empty"),
                         "ready" if s["audio_resolver"]=="ready" else ("pending" if s["audio_resolver"]=="pending" else "empty")),
                    unsafe_allow_html=True)
        if audio_csv.exists():
            st.caption(f"`{audio_csv.name}`")
            if norm_csv.exists():
                cov = compute_audio_coverage(norm_csv, audio_csv, use_stem_fallback=st.session_state.get("use_stem_fallback", True))
                det_pct = (cov.matched_rows / cov.total_rows * 100.0) if cov.total_rows else 0.0
                file_pct = (cov.matched_unique_files / cov.total_unique_files * 100.0) if cov.total_unique_files else 0.0
                st.caption(f"Coverage — Detections: **{det_pct:.1f}%**; Files: **{file_pct:.1f}%**")
        if st.button("Open Audio mapping", key="open_step2"):
            st.session_state.route = "locate_audio"; st.rerun()

    with step3:
        st.markdown("#### 3) Metadata mapping")
        st.markdown(chip("ready" if s["metadata_joins"]=="ready" else ("pending" if s["metadata_joins"]=="pending" else "empty"),
                         "ready" if s["metadata_joins"]=="ready" else ("pending" if s["metadata_joins"]=="pending" else "empty")),
                    unsafe_allow_html=True)
        if enrich_csv.exists(): st.caption(f"`{enrich_csv.name}`")
        if st.button("Open Metadata mapping", key="open_step3"):
            st.session_state.route = "metadata"; st.rerun()

    st.divider()
    st.write("### Import stats")
    stats = compute_import_stats(norm_csv, audio_csv if audio_csv.exists() else None, enrich_csv if enrich_csv.exists() else None, use_stem_fallback=st.session_state.get("use_stem_fallback", True))
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Detections (rows)", f"{stats['detections_rows']:,}")
    c2.metric("Unique files in detections", f"{stats['unique_files_in_detections']:,}")
    c3.metric("Audio files indexed", f"{stats['audio_files_indexed']:,}")
    c4.metric("Detections with audio", f"{stats['detections_with_audio']:,}")
    c5.metric("Metadata rows", f"{stats['metadata_join_rows']:,}")
    c6.metric("Final importable rows", f"{stats['final_rows']:,}")

    st.divider()
    all_ready = (s["import_results"]=="ready" and s["audio_resolver"]=="ready" and s["metadata_joins"]=="ready")
    if all_ready:
        st.success("All three import steps are complete.")
        if st.button("Launch PAMalytics dashboard", key="launch_dashboard"):
            st.session_state.route = "dashboard"; st.rerun()
    else:
        st.info("Complete all three steps above to launch the PAMalytics dashboard.")

    st.divider()
    cols = st.columns(3)
    with cols[0]:
        st.write("**IDs**"); st.code(f"project_id: {data['project_id']}"); st.code(f"created_by: {data.get('created_by','user')}")
    with cols[1]:
        st.write("**Dates**"); st.code(f"created_at: {data.get('created_at','')}"); st.code(f"last_opened: {data.get('last_opened','')}")
    with cols[2]:
        st.write("**Folders**"); 
        for k, v in (data.get("paths") or {}).items(): st.code(f"{k}: {v}")
    st.divider()
    nav_row("Back to Project Hub", "hub", key_prefix="overview_bottom")

def _auto_guess(colnames: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

# ------------------ Import results (Data mapping) ------------------
def view_import_results() -> None:
    import pandas as pd
    if not st.session_state.get("auth_user"): st.session_state.route = "login"; st.rerun()
    if not st.session_state.get("current_project"): st.session_state.route = "hub"; st.rerun()

    proj_path = Path(st.session_state.current_project)
    st.title("Import results — Data mapping")
    st.caption("Upload classifier results, map columns, validate (edit if needed), and save a normalised copy.")
    nav_row("Back to Overview", "overview", "Go to Audio mapping", "locate_audio", key_prefix="import_top")

    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    if norm_csv.exists():
        st.success(f"Detected existing normalised data: `{norm_csv}`")
        with st.expander("Preview saved normalised data", expanded=False):
            try:
                prev_df = pd.read_csv(norm_csv)
                st.dataframe(prev_df.head(50), use_container_width=True)
                st.caption(f"Rows: {len(prev_df)}")
            except Exception as e:
                st.error(f"Could not read saved file: {e}")
        st.divider()

    up = st.file_uploader("Upload CSV / TSV / Parquet", type=["csv", "tsv", "parquet"], key="import_uploader")
    if up is None and st.session_state.get("import_params", {}).get("df") is None and not norm_csv.exists():
        st.info("Select a file to begin."); return

    if up is not None:
        try:
            if up.name.endswith(".parquet"):
                df = pd.read_parquet(up)
            else:
                try: df = pd.read_csv(up)
                except Exception: up.seek(0); df = pd.read_csv(up, sep="\t")
        except Exception as e:
            st.error(f"Could not read file: {e}"); return
        if df is None or df.empty:
            st.error("No rows found."); return
        st.session_state.import_params["filename"] = up.name
        st.session_state.import_params["df"] = df

    df = st.session_state.import_params.get("df")
    if df is None:
        st.info("No uploaded data in this session. You can preview the saved normalised data above, or upload a new file.")
        return

    st.write("Preview (first 20 rows):")
    st.dataframe(df.head(20), use_container_width=True)

    cols = list(df.columns)
    st.subheader("1) Map required columns")

    file_guess  = _auto_guess(cols, ["filename","file","filepath","path","source_file"])
    start_guess = _auto_guess(cols, ["start","start_s","start_time_s","begin","onset","start_sec","start_time"])
    end_guess   = _auto_guess(cols, ["end","end_s","end_time_s","offset","end_sec","end_time","duration","duration_s"])
    score_guess = _auto_guess(cols, ["score","confidence","prob","probability","p_hat"])
    time_guess  = _auto_guess(cols, ["timestamp_utc","timestamp","datetime","date_time","utc"])

    def pick(name_key: str, label: str, default: Optional[str]):
        current = st.session_state.import_params.get(name_key, "—")
        if current == "—" and default in cols: current = default
        idx = (cols.index(current) + 1) if (current in cols) else 0
        choice = st.selectbox(label, ["—"] + cols, index=idx, key=f"select_{name_key}")
        st.session_state.import_params[name_key] = choice
        return choice

    source_file_col = pick("source_file", "File path / name → `source_file`", file_guess)
    start_col       = pick("start_s", "Start time (seconds) → `start_s`", start_guess)
    end_col         = pick("end_s",   "End time (seconds or duration) → `end_s`", end_guess)
    score_col       = pick("score",   "Confidence [0,1] → `score` (optional)", score_guess)
    ts_col          = pick("timestamp_utc", "Timestamp (UTC or datetime) → `timestamp_utc` (optional)", time_guess)

    # Label source
    st.subheader("2) Choose how to derive the label (present/absent)")
    label_mode = st.session_state.import_params.get("label_mode", "binary_presence_column")
    label_mode = st.radio(
        "Label source",
        options=["binary_presence_column", "use_label_column"],
        index=0 if label_mode == "binary_presence_column" else 1,
        format_func=lambda x: "Binary presence column (0/1, True/False, yes/no…)" if x=="binary_presence_column" else "Use existing label column",
        horizontal=False,
        key="label_mode_radio"
    )
    st.session_state.import_params["label_mode"] = label_mode

    presence_col = st.session_state.import_params.get("presence_col", "—")
    positive_tokens = st.session_state.import_params.get("positive_tokens", "1,true,yes,present,y,t")
    positive_label_name = st.session_state.import_params.get("positive_label_name", "present")
    keep_only_present = st.session_state.import_params.get("keep_only_present", True)
    label_col = st.session_state.import_params.get("label_col", "—")
    canonicalise_existing = st.session_state.import_params.get("canonicalise_existing", False)
    present_value_for_existing = st.session_state.import_params.get("present_value_for_existing", "1")

    if label_mode == "binary_presence_column":
        guess_presence = _auto_guess(cols, ["present","presence","detected","label","class","species"])
        if presence_col == "—" and guess_presence in cols: presence_col = guess_presence
        presence_col = st.selectbox("Presence column", ["—"] + cols,
                                    index=(cols.index(presence_col)+1) if presence_col in cols else 0,
                                    key="presence_col_select")
        positive_tokens = st.text_input("Values that mean 'present' (comma-separated)", value=positive_tokens, key="positive_tokens")
        positive_label_name = st.text_input("Canonical label for present detections", value=positive_label_name, key="positive_label_name")
        keep_only_present = st.checkbox("Keep only present rows (detections only)", value=bool(keep_only_present), key="keep_only_present")
    else:
        label_col_guess = _auto_guess(cols, ["label","species","class","prediction"])
        if label_col == "—" and label_col_guess in cols: label_col = label_col_guess
        label_col = st.selectbox("Label column", ["—"] + cols,
                                 index=(cols.index(label_col)+1) if label_col in cols else 0,
                                 key="label_col_select")
        canonicalise_existing = st.checkbox("Canonicalise this label to present/absent", value=bool(canonicalise_existing), key="canonicalise_existing")
        present_value_for_existing = st.text_input("Value that means 'present' (used only when canonicalising)", value=str(present_value_for_existing), key="present_value_for_existing")

    st.session_state.import_params.update({
        "presence_col": presence_col,
        "positive_tokens": positive_tokens,
        "positive_label_name": positive_label_name,
        "keep_only_present": bool(keep_only_present),
        "label_col": label_col,
        "canonicalise_existing": bool(canonicalise_existing),
        "present_value_for_existing": present_value_for_existing,
    })

    # Required fields
    missing = []
    if source_file_col == "—": missing.append("source_file")
    if start_col == "—":       missing.append("start_s")
    if end_col == "—":         missing.append("end_s")
    if label_mode == "binary_presence_column" and (presence_col in (None, "—")): missing.append("presence column")
    if label_mode == "use_label_column" and (label_col in (None, "—")): missing.append("label column")
    if missing: st.warning("Please map required fields: " + ", ".join(missing))

    st.subheader("3) Options")
    convert_ms = st.checkbox("Times are in milliseconds (convert to seconds)", value=bool(st.session_state.import_params.get("convert_ms", False)), key="convert_ms")
    assume_utc = st.checkbox("Interpret `timestamp_utc` as UTC when timezone is missing", value=bool(st.session_state.import_params.get("assume_utc", True)), key="assume_utc")
    st.session_state.import_params.update({"convert_ms": bool(convert_ms), "assume_utc": bool(assume_utc)})

    # Build preview
    st.markdown("### 4) Build preview (editable)")
    disabled = bool(missing)
    if _btn("Build preview", key="build_preview_btn") and not disabled:
        norm, notes = _build_normalised_table(
            df=df, source_file_col=source_file_col, start_col=start_col, end_col=end_col,
            score_col=score_col, ts_col=ts_col, convert_ms=bool(convert_ms), assume_utc=bool(assume_utc),
            label_mode=label_mode, presence_col=presence_col, positive_tokens=positive_tokens,
            positive_label_name=positive_label_name, keep_only_present=bool(keep_only_present),
            label_col=label_col, canonicalise_existing=bool(canonicalise_existing),
            present_value_for_existing=present_value_for_existing,
        )
        st.session_state.import_preview_ready = True
        st.session_state.import_preview_df = norm.to_dict(orient="records")
        st.session_state.import_last_saved = None
        st.session_state.import_notes = notes
        st.rerun()
    if disabled: st.caption("Map all required fields above to enable the preview.")

    # Render preview
    if st.session_state.get("import_preview_ready") and isinstance(st.session_state.get("import_preview_df"), list):
        import pandas as pd
        norm = pd.DataFrame.from_records(st.session_state.import_preview_df)
        for n in st.session_state.get("import_notes", []): st.caption(n)
        if norm.empty:
            st.warning("No rows to display. Please check that your **present** value(s) are correct for the selected presence column.")
            return

        issues = []
        if (norm["end_s"] < norm["start_s"]).any(): issues.append("Some rows have end_s < start_s.")
        if norm["start_s"].isna().any() or norm["end_s"].isna().any(): issues.append("Some rows have invalid start/end times.")
        if norm["source_file"].isna().any() or (norm["source_file"].astype(str).str.strip() == "").any(): issues.append("Some rows have missing source_file.")
        for msg in issues: st.error(msg)

        st.subheader("5) Validate & edit the final mapped data")
        edited = st.data_editor(norm, use_container_width=True, num_rows="dynamic", key="norm_editor")

        if _btn("Save normalised copy", key="save_norm_btn"):
            out_dir = project_path(proj_path, "data_normalised"); out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "detections_normalised.csv"; edited.to_csv(out_csv, index=False)
            manifest = {
                "adapter":"auto",
                "mapping":{
                    "source_file": source_file_col,
                    "start_s": start_col,
                    "end_s": end_col,
                    "label": (f"{presence_col}→present/absent" if label_mode=="binary_presence_column"
                              else label_col if not st.session_state.import_params.get("canonicalise_existing")
                              else f"{label_col}→present/absent [{st.session_state.import_params.get('present_value_for_existing')}=present]"),
                    "score": None if score_col == "—" else score_col,
                    "timestamp_utc": None if ts_col == "—" else ts_col,
                },
                "options":{
                    "convert_ms": bool(convert_ms), "assume_utc": bool(assume_utc),
                    "label_mode": label_mode,
                    "positive_tokens": st.session_state.import_params.get("positive_tokens") if label_mode=="binary_presence_column" else None,
                    "keep_only_present": bool(st.session_state.import_params.get("keep_only_present")) if label_mode=="binary_presence_column" else None,
                    "canonicalise_existing": bool(st.session_state.import_params.get("canonicalise_existing")) if label_mode=="use_label_column" else None,
                    "present_value_for_existing": st.session_state.import_params.get("present_value_for_existing") if label_mode=="use_label_column" else None,
                },
                "input_file": st.session_state.import_params.get("filename"),
                "rows_out": int(len(edited)),
                "created_at": datetime.now(dt_timezone.utc).isoformat(),
                "app_version": "pamalytics_studio 0.3.0",
            }
            ws_dir = project_path(proj_path, "workspace"); ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / "ingest_mapping.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            st.session_state.import_last_saved = str(out_csv); set_status(proj_path, "import_results", "ready")
            st.success(f"Saved normalised detections to: `{out_csv}`")

        if st.session_state.get("import_last_saved"):
            nav_row("Back to Overview", "overview", "Go to Audio mapping", "locate_audio", key_prefix="import_after_save")

# ---------- Normalisation builder ----------
def _build_normalised_table(
    df, source_file_col: str, start_col: str, end_col: str, score_col: Optional[str],
    ts_col: Optional[str], convert_ms: bool, assume_utc: bool, label_mode: str,
    presence_col: Optional[str], positive_tokens: str, positive_label_name: str,
    keep_only_present: bool, label_col: Optional[str], canonicalise_existing: bool,
    present_value_for_existing: str,
):
    import pandas as pd
    from datetime import datetime as _dt

    def to_seconds(series):
        s = pd.to_numeric(series, errors="coerce")
        if convert_ms: s = s / 1000.0
        return s

    # Subset based on label
    if label_mode == "binary_presence_column":
        tokens = {t.strip().lower() for t in positive_tokens.split(",") if t.strip() != ""}
        ser = df[presence_col].astype(str).str.strip().str.lower()
        present_mask = ser.isin(tokens)
        if keep_only_present:
            if not present_mask.any():
                empty = pd.DataFrame(columns=["source_file","start_s","end_s","label","score","timestamp_utc"])
                return empty, ["No rows matched the chosen present value(s)."]
            base = df.loc[present_mask].copy()
        else:
            base = df.copy()
    else:
        base = df.copy()

    norm = pd.DataFrame()
    norm["source_file"] = base[source_file_col].astype(str)

    # End may be duration
    if end_col.lower() in {"duration", "duration_s"}:
        start_vals = to_seconds(base[start_col]); dur_vals = to_seconds(base[end_col])
        norm["start_s"] = start_vals; norm["end_s"] = start_vals + dur_vals
    else:
        norm["start_s"] = to_seconds(base[start_col]); norm["end_s"] = to_seconds(base[end_col])

    # Labels
    if label_mode == "binary_presence_column":
        if keep_only_present:
            norm["label"] = positive_label_name or "present"
        else:
            ser_b = base[presence_col].astype(str).str.strip().str.lower()
            tokens_b = {t.strip().lower() for t in positive_tokens.split(",") if t.strip() != ""}
            present_mask_b = ser_b.isin(tokens_b)
            norm["label"] = (positive_label_name or "present")
            norm.loc[~present_mask_b, "label"] = "absent"
    else:
        lbl = base[label_col].astype(str)
        if canonicalise_existing:
            pv = str(present_value_for_existing).strip().lower()
            mask = lbl.str.strip().str.lower().eq(pv)
            lbl = lbl.mask(mask, "present"); lbl = lbl.where(lbl == "present", "absent")
        norm["label"] = lbl

    # Score
    norm["score"] = pd.to_numeric(base[score_col], errors="coerce") if (score_col and score_col != "—") else pd.NA

    # Timestamp parse (incl. 20250513_083937)
    notes = []
    if ts_col and ts_col != "—":
        raw = base[ts_col].astype(str).str.strip()
        ts = pd.to_datetime(raw, errors="coerce", utc=False)
        if ts.isna().mean() > 0.2:
            def parse_custom(x: str):
                x = x.strip()
                try:
                    if len(x) == 15 and x[8] == "_" and x[:8].isdigit() and x[9:].isdigit():
                        return _dt.strptime(x, "%Y%m%d_%H%M%S")
                    if len(x) == 14 and x.isdigit():
                        return _dt.strptime(x, "%Y%m%d%H%M%S")
                except Exception:
                    return None
                return None
            parsed = raw.apply(parse_custom)
            ts = ts.where(~ts.isna(), pd.to_datetime(parsed, errors="coerce", utc=False))
        try:
            tzinfo = ts.dt.tz
        except Exception:
            tzinfo = None
        if tzinfo is None:
            if assume_utc:
                try:
                    ts = ts.dt.tz_localize("UTC"); notes.append("Naïve datetimes interpreted as UTC (+00:00).")
                except Exception:
                    pass
            else:
                notes.append("Datetimes are timezone-naïve; consider enabling ‘Interpret as UTC’.")
        else:
            try:
                ts = ts.dt.tz_convert("UTC"); notes.append("Datetimes converted to UTC (+00:00).")
            except Exception:
                pass
        norm["timestamp_utc"] = ts

    return norm, notes

# ------------------ Audio mapping ------------------
def view_locate_audio() -> None:
    import pandas as pd
    if not st.session_state.get("auth_user"): st.session_state.route = "login"; st.rerun()
    if not st.session_state.get("current_project"): st.session_state.route = "hub"; st.rerun()

    proj_path = Path(st.session_state.current_project); ensure_paths_schema(proj_path)

    st.title("Locate your audio files — Audio mapping")
    st.caption("Point to one or more folders. We'll index audio files recursively and create a filename → absolute path map.")
    nav_row("Back to Overview", "overview", "Go to Metadata mapping", "metadata", key_prefix="locate_top")

    col_pick, col_hint = st.columns([1,2])
    if col_pick.button("Browse for a folder…", key="browse_folder"):
        chosen = pick_folder_dialog()
        if chosen: st.session_state.audio_dirs = sorted(set(st.session_state.audio_dirs + [chosen]))
        else: st.warning("No folder selected (or folder chooser not available).")
    col_hint.caption("Tip: you can add multiple folders; duplicates are handled below.")

    with st.form("audio_dirs_form", clear_on_submit=False):
        dirs_text = st.text_area("Folders to scan (one per line)",
                                 value="\n".join(st.session_state.audio_dirs) if st.session_state.audio_dirs else "",
                                 placeholder="/Volumes/drive/site_a\n/Users/you/recordings/batch1",
                                 height=120)
        exts = st.text_input("File extensions to include (comma-separated)", value=".wav,.flac,.mp3")
        submitted = st.form_submit_button("Scan folders")

    if submitted:
        st.session_state.audio_dirs = [d.strip() for d in dirs_text.splitlines() if d.strip()]
        include_exts = {e.strip().lower() if e.strip().startswith(".") else ("." + e.strip().lower())
                        for e in exts.split(",") if e.strip()}
        files = []
        for d in st.session_state.audio_dirs:
            p = Path(d).expanduser()
            if not p.exists():
                st.error(f"Folder not found: {p}"); continue
            for root, _, names in os.walk(p):
                for nm in names:
                    ext = os.path.splitext(nm)[1].lower()
                    if ext in include_exts:
                        full = Path(root) / nm
                        try: mtime = full.stat().st_mtime
                        except Exception: mtime = None
                        files.append({"filename": nm, "path": str(full), "ext": ext, "mtime": mtime})
        if not files:
            st.warning("No audio files found. Check folders and extensions."); st.session_state.audio_map_df = None
        else:
            df = pd.DataFrame(files)
            df["duplicate_name"] = df["filename"].duplicated(keep=False)
            st.session_state.audio_map_df = df.to_dict(orient="records")

    if isinstance(st.session_state.audio_map_df, list):
        df = pd.DataFrame.from_records(st.session_state.audio_map_df)
        st.write(f"Indexed **{len(df)}** audio files from **{len(st.session_state.audio_dirs)}** folder(s).")
        if "duplicate_name" in df.columns and df["duplicate_name"].any():
            st.warning(f"Found {int(df['duplicate_name'].sum())} files with duplicate names across different folders.")

        with st.expander("Preview indexed files", expanded=False):
            st.dataframe(df[["filename","path","ext"]].head(200), use_container_width=True)

        st.subheader("Disambiguation for duplicate filenames")
        rule = st.radio(
            "If the same filename exists in multiple folders, keep:",
            options=["most_recent_mtime", "oldest_mtime", "first_seen", "all_copies"],
            index=0,
            format_func=lambda x: {
                "most_recent_mtime":"The most recently modified file",
                "oldest_mtime":"The oldest modified file",
                "first_seen":"The first one found (folder order)",
                "all_copies":"Keep all copies (results will have duplicate filenames)",
            }[x],
            key="audio_rule_radio"
        )

        if _btn("Build mapping preview", key="build_audio_map_btn"):
            mapping = build_audio_mapping(pd.DataFrame.from_records(st.session_state.audio_map_df), rule)
            st.session_state.audio_map_preview = mapping.to_dict(orient="records"); st.rerun()

        if isinstance(st.session_state.get("audio_map_preview"), list):
            mapping = pd.DataFrame.from_records(st.session_state.audio_map_preview)
            st.subheader("Mapping preview")
            st.dataframe(mapping.head(200), use_container_width=True)
            st.caption(f"Rows: {len(mapping)}  • Columns: filename, path")

            use_stem = st.checkbox(
                "Allow stem fallback (match without extension when filenames differ only by extension)",
                value=st.session_state.get("use_stem_fallback", True),
                key="use_stem_fallback"
            )
            norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
            if norm_csv.exists():
                cov = compute_audio_coverage(norm_csv, mapping, use_stem_fallback=use_stem)
                det_pct = (cov.matched_rows / cov.total_rows * 100.0) if cov.total_rows else 0.0
                file_pct = (cov.matched_unique_files / cov.total_unique_files * 100.0) if cov.total_unique_files else 0.0
                st.info(
                    f"Detections coverage: **{det_pct:.1f}%** ({cov.matched_rows:,} / {cov.total_rows:,})  •  "
                    f"File coverage: **{file_pct:.1f}%** ({cov.matched_unique_files:,} / {cov.total_unique_files:,})"
                )

            if _btn("Save mapping", key="save_audio_map_btn"):
                out = project_path(proj_path, "workspace"); out.mkdir(parents=True, exist_ok=True)
                out_csv = out / "audio_paths.csv"; mapping.to_csv(out_csv, index=False)
                st.session_state.audio_save_path = str(out_csv); set_status(proj_path, "audio_resolver", "ready")
                st.success(f"Saved audio mapping to: `{out_csv}`")

                if norm_csv.exists():
                    cov = compute_audio_coverage(norm_csv, out_csv, use_stem_fallback=st.session_state.get("use_stem_fallback", True))
                    det_pct = (cov.matched_rows / cov.total_rows * 100.0) if cov.total_rows else 0.0
                    file_pct = (cov.matched_unique_files / cov.total_unique_files * 100.0) if cov.total_unique_files else 0.0
                    st.info(
                        f"Final coverage — Detections: **{det_pct:.1f}%** ({cov.matched_rows:,} / {cov.total_rows:,});  "
                        f"Files: **{file_pct:.1f}%** ({cov.matched_unique_files:,} / {cov.total_unique_files:,})"
                    )

    nav_row("Back to Overview", "overview", "Go to Metadata mapping", "metadata", key_prefix="locate_bottom")

def pick_folder_dialog() -> Optional[str]:
    """Open a native folder-picker dialog and return the chosen absolute path, or None."""
    try:
        system = platform.system().lower()
        if "darwin" in system or "mac" in system:
            script = 'set _folder to POSIX path of (choose folder with prompt "Select an audio folder")\nreturn _folder'
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            path = res.stdout.strip(); return path or None
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title="Select an audio folder")
            root.destroy(); return folder or None
    except Exception:
        return None

def build_audio_mapping(df, rule: str):
    import pandas as pd
    if rule == "all_copies": return df[["filename","path"]].copy()
    df2 = df.copy()
    if rule == "most_recent_mtime": df2 = df2.sort_values(by=["filename","mtime"], ascending=[True, False])
    elif rule == "oldest_mtime":    df2 = df2.sort_values(by=["filename","mtime"], ascending=[True, True])
    elif rule == "first_seen":
        df2 = df2.sort_values(by=["filename"]).drop_duplicates(subset=["filename"], keep="first")
        return df2[["filename","path"]].copy()
    df2 = df2.drop_duplicates(subset=["filename"], keep="first")
    return df2[["filename","path"]].copy()

# ------------------ Metadata join ------------------
def view_metadata() -> None:
    import pandas as pd, os
    if not st.session_state.get("auth_user"): st.session_state.route = "login"; st.rerun()
    if not st.session_state.get("current_project"): st.session_state.route = "hub"; st.rerun()

    proj_path = Path(st.session_state.current_project)
    st.title("Metadata mapping — Join metadata")
    st.caption("Upload a metadata table (e.g., recorder/site/lat/lon). Map a join key and save enriched detections.")
    nav_row("Back to Audio mapping", "locate_audio", "Back to Overview", "overview", key_prefix="meta_top")

    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    if not norm_csv.exists():
        st.error("No normalised detections found. Please complete Data mapping first."); return

    det = pd.read_csv(norm_csv)
    if det.empty:
        st.error("Detections table is empty."); return
    det["basename"] = det["source_file"].astype(str).apply(lambda p: os.path.basename(p))
    det["stem"] = det["basename"].str.replace(r"\.[^.]+$", "", regex=True)
    det["recorder_id"] = det["basename"].apply(lambda n: n.split("_", 1)[0] if "_" in n else n)

    with st.expander("Preview derived columns", expanded=False):
        st.dataframe(det[["source_file","basename","recorder_id"]].head(20), use_container_width=True)

    st.subheader("1) Upload metadata table")
    up = st.file_uploader("Upload metadata CSV / TSV / Parquet", type=["csv","tsv","parquet"], key="meta_up")
    if up is None:
        st.info("Select a metadata file to begin."); return

    try:
        if up.name.endswith(".parquet"): meta = pd.read_parquet(up)
        else:
            try: meta = pd.read_csv(up)
            except Exception: up.seek(0); meta = pd.read_csv(up, sep="\t")
    except Exception as e:
        st.error(f"Could not read metadata: {e}"); return
    if meta.empty:
        st.error("Metadata file is empty."); return

    st.subheader("2) Choose join keys")
    det_key = st.selectbox("Detections key", options=["recorder_id","basename","stem","source_file"], index=0)
    meta_cols = list(meta.columns)
    meta_key = st.selectbox("Metadata key (column in uploaded table)", options=meta_cols)

    st.subheader("3) Preview join")
    merged = det.merge(meta, left_on=det_key, right_on=meta_key, how="left")
    st.dataframe(merged.head(200), use_container_width=True)
    join_rate = 100.0 * (1.0 - float(merged[meta_key].isna().mean()))
    st.info(f"Join coverage: **{join_rate:.1f}%** of detections joined with metadata (via `{det_key} == {meta_key}`).")

    if _btn("Save enriched detections"):
        out_csv = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
        merged.to_csv(out_csv, index=False)
        set_status(proj_path, "metadata_joins", "ready")
        st.success(f"Saved enriched detections to: `{out_csv}`")
        nav_row("Back to Overview", "overview", "Launch dashboard", "dashboard", key_prefix="meta_after_save")

# ------------------ Dashboard (integrated PAMalytics) ------------------
def _unhide_chrome_for_pa():
    st.markdown(
        """
        <style>
          header, footer, [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: block !important; visibility: visible !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _load_renderer(module_stem: str, func_name: str):
    """
    Resolve a renderer function from:
      1) scripts/ as a module import (e.g., 'dashboard')
      2) scripts/*.py by filename (case-insensitive)
      3) scripts/pages/*.py by filename (case-insensitive), including '1_Validate.py'
    """
    import importlib, importlib.util

    # 1) normal import
    for cand in {module_stem, module_stem.capitalize()}:
        try:
            mod = importlib.import_module(cand)
            fn = getattr(mod, func_name, None)
            if callable(fn): return fn
        except ModuleNotFoundError:
            pass

    def _scan_and_load(folder: Path, stem: str):
        for p in folder.glob("*.py"):
            if p.stem.lower() == stem.lower():
                spec = importlib.util.spec_from_file_location(p.stem, p)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
                    fn = getattr(mod, func_name, None)
                    if callable(fn): return fn
        return None

    # 2) scripts/
    fn = _scan_and_load(SCRIPTS_DIR, module_stem)
    if fn: return fn

    # 3) scripts/pages/ (support names like 1_Validate.py)
    pages_dir = SCRIPTS_DIR / "pages"
    if pages_dir.exists():
        fn = _scan_and_load(pages_dir, module_stem)
        if not fn:
            for p in pages_dir.glob("*.py"):
                if module_stem.lower() in p.stem.lower():
                    spec = importlib.util.spec_from_file_location(p.stem, p)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
                        fn = getattr(mod, func_name, None)
                        if callable(fn): return fn
    return None

def _make_sources(proj_path: Path) -> Dict[str, Any]:
    audio_csv = project_path(proj_path, "workspace") / "audio_paths.csv"
    base_dir = str(project_path(proj_path, "root"))
    return {
        "project_root": str(proj_path),
        "audio_map_csv": str(audio_csv) if audio_csv.exists() else None,
        "base_dir": base_dir,
        "use_stem_fallback": st.session_state.get("use_stem_fallback", True),
    }

def view_dashboard() -> None:
    _unhide_chrome_for_pa()
    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"; st.rerun()
    if not st.session_state.get("current_project"):
        st.session_state.route = "hub"; st.rerun()

    proj_path = Path(st.session_state.current_project)
    st.title("PAMalytics")
    st.caption("Studio-imported data via existing PAMalytics pages.")
    nav_row("Back to Overview", "overview", key_prefix="pa_top")

    df, notes = build_analysis_dataset(proj_path, use_stem_fallback=st.session_state.get("use_stem_fallback", True))
    if df is None or df.empty:
        st.error("; ".join(notes) if notes else "No data to display."); return

    sources = _make_sources(proj_path)

    fn_dash = _load_renderer("dashboard", "render_dashboard")
    if fn_dash:
        fn_dash(df, sources)
    else:
        st.error("Could not find `render_dashboard(df, sources)` in scripts/ or scripts/pages/.")
        return

    with st.expander("Open other PAMalytics pages", expanded=False):
        col1, col2, col3 = st.columns(3)
        if col1.button("Validation"):
            fn = _load_renderer("validation", "render_validation")
            if fn: fn(df, sources)
            else: st.error("`render_validation(df, sources)` not found.")
        if col2.button("Settings"):
            fn = _load_renderer("settings", "render_settings")
            if fn: fn(df, sources)
            else: st.error("`render_settings(df, sources)` not found.")
        if col3.button("Occupancy"):
            fn = _load_renderer("occupancy", "render_occupancy")
            if fn: fn(df, sources)
            else: st.error("`render_occupancy(df, sources)` not found.")

    nav_row("Back to Overview", "overview", key_prefix="pa_bottom")

# ---------- Router ----------
def _route_from_query():
    # New API: st.query_params
    qp = dict(st.query_params)
    return qp.get("route")

# If route supplied in URL, respect it before drawing
forced = _route_from_query()
if forced and forced != st.session_state.get("route"):
    st.session_state["route"] = forced
    st.rerun()

route = st.session_state.get("route", "login")
if route == "login":          view_login()
elif route == "hub":          view_hub()
elif route == "overview":     view_overview()
elif route == "import":       view_import_results()
elif route == "locate_audio": view_locate_audio()
elif route == "metadata":     view_metadata()
elif route == "dashboard":    view_dashboard()
else:
    st.session_state.route = "login"; st.rerun()
