@echo off
setlocal ENABLEDELAYEDEXPANSION

REM One-click end-to-end pipeline: download -> process -> analyze -> train -> test -> backtest -> report

IF NOT EXIST .\venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv venv || (
    echo Failed to create virtualenv. Ensure Python is installed.
    pause
    exit /b 1
  )
)

echo Updating pip and installing requirements...
.\venv\Scripts\python -m pip install -U pip >nul 2>&1
.\venv\Scripts\python -m pip install -r requirements.txt || (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo Running pipeline...
.\venv\Scripts\python -m simple_app.run_all

echo.
echo Finished. Press any key to close.
pause >nul

