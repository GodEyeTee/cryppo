@echo off
REM Simple one-click launcher for the simple_app menu
IF EXIST .\venv\Scripts\python.exe (
  .\venv\Scripts\python -m simple_app
  GOTO :eof
)

python -m simple_app

