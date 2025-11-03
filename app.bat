@echo off
REM Activate conda if available; fall back to pip env
IF EXIST "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
)

cd /d "%~dp0"
streamlit run app.py