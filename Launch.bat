@echo off
cd /d "%~dp0"
start "GreenToBuy API" cmd /k "python -m uvicorn server:app --reload --port 8000"
start "GreenToBuy Dashboard" cmd /k "python -m streamlit run app.py"
