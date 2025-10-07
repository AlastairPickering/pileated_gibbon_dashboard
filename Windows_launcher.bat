@echo off
cd /d "%~dp0"
REM Run the Python bootstrapper (it creates venv, installs requirements, and launches Streamlit)
py -3 scripts/launch_dashboard.py || python scripts/launch_dashboard.py