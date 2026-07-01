@echo off
setlocal enabledelayedexpansion

echo ==================================
echo  Agent App - Quick Start
echo ==================================

set "APP_DIR=%~dp0"

REM --------------------------------------------------
REM 1. Check Python version
REM --------------------------------------------------
set "PYTHON_BIN="
where python 2>nul >nul
if %errorlevel% equ 0 (
    python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
    if !errorlevel! equ 0 set "PYTHON_BIN=python"
)

if "%PYTHON_BIN%"=="" (
    echo ERROR: Python ^>= 3.10 is required but not found.
    echo   Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('%PYTHON_BIN% --version 2^>^&1') do echo [^|] Python %%v

REM --------------------------------------------------
REM 2. Check / install Poetry
REM --------------------------------------------------
where poetry 2>nul >nul
if %errorlevel% equ 0 (
    for /f "tokens=3" %%v in ('poetry --version 2^>^&1') do echo [^|] Poetry %%v
) else (
    echo [...] Installing Poetry...
    %PYTHON_BIN% -m pip install --quiet poetry
    REM Add pip user Scripts dir to PATH so the freshly installed poetry is found
    for /f %%i in ('%PYTHON_BIN% -c "import site; print(site.USER_SITE)" 2^>nul') do set "USER_SITE=%%i"
    if defined USER_SITE set "PATH=%USER_SITE%\..\Scripts;%PATH%"
)

REM --------------------------------------------------
REM 3. Start dependencies (PostgreSQL + Qdrant) with Docker
REM --------------------------------------------------
where docker 2>nul >nul
if %errorlevel% equ 0 (
    where docker-compose 2>nul >nul
    if !errorlevel! equ 0 (
        echo.
        echo ^=^>^> Starting PostgreSQL and Qdrant...
        cd /d "%APP_DIR%"
        docker compose up -d ai_agent_db ai_agent_qdrant
        echo [^|] Dependencies are running
    )
) else (
    echo [!] Docker not found - skipping dependency startup.
    echo     Make sure PostgreSQL (pgvector) and Qdrant are running manually.
)

REM --------------------------------------------------
REM 4. Install project dependencies
REM --------------------------------------------------
echo.
echo ^=^>^> Installing project dependencies...
poetry install --no-interaction --no-ansi --quiet

REM --------------------------------------------------
REM 5. Run the application
REM --------------------------------------------------
echo.
echo ^=^>^> Starting Agent App...
echo     App:       http://localhost:2424
echo     API Docs:  http://localhost:2424/api-doc
echo.
cd /d "%APP_DIR%"
set "PYTHONPATH=%APP_DIR%src"
poetry run python src/agent_app/main.py

pause
