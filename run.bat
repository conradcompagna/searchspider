@echo off
rem Konbaung Search: first run creates .venv and installs requirements; then starts the server and opens the page.
rem The server restarts itself if it stops; its output goes to logs\server.log. Close this window to stop it.
cd /d "%~dp0"
if not exist logs mkdir logs
echo %date% %time% run.bat started >> logs\run.log
rem Stop any earlier server still holding the port, so the page never talks to an old version.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765 " ^| findstr LISTENING') do (
  echo Stopping earlier server, process %%p
  taskkill /F /PID %%p >nul 2>&1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || goto :err
  .venv\Scripts\python -m pip install --upgrade pip >nul
  .venv\Scripts\python -m pip install -r requirements.txt || goto :err
)
if not exist logs mkdir logs
rem Install new analyzer dependency for environments created before English-only search.
.venv\Scripts\python -c "import snowballstemmer" >nul 2>&1
if errorlevel 1 .venv\Scripts\python -m pip install -r requirements.txt || goto :err
if not defined OPEN_BROWSER set OPEN_BROWSER=1
set PYTHONUNBUFFERED=1
:loop
echo %date% %time% starting server; output in logs\server.log
echo ==== %date% %time% server start ==== >> logs\server.log
.venv\Scripts\python backend\app.py >> logs\server.log 2>&1
echo %date% %time% server stopped; restarting in 5 s (close this window to stop)
echo ==== %date% %time% server stopped, exit code %errorlevel% ==== >> logs\server.log
set OPEN_BROWSER=0
timeout /t 5 /nobreak >nul
goto loop
:err
echo Setup failed. Check that Python 3.10+ is installed and on PATH.
pause
