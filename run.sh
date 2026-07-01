#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo " Agent App — Quick Start"
echo "=================================="

# --------------------------------------------------
# 1. Check Python version
# --------------------------------------------------
REQUIRED_PYTHON="3.10"
PYTHON_BIN=""

for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        if [[ "$(printf '%s\n' "$REQUIRED_PYTHON" "$ver" | sort -V | head -1)" == "$REQUIRED_PYTHON" ]]; then
            PYTHON_BIN="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Python >= $REQUIRED_PYTHON is required but not found."
    echo "  Install it from https://www.python.org/downloads/"
    exit 1
fi
echo "[✓] Python $($PYTHON_BIN --version | cut -d' ' -f2)"

# --------------------------------------------------
# 2. Check / install Poetry
# --------------------------------------------------
if ! command -v poetry &>/dev/null; then
    echo "[...] Installing Poetry..."
    $PYTHON_BIN -m pip install --quiet poetry
    # Add user-base bin dir to PATH for freshly installed binaries
    export PATH="$($PYTHON_BIN -m site --user-base 2>/dev/null || echo "$HOME/.local")/bin:$PATH"
fi

POETRY_VER="$(poetry --version 2>/dev/null | cut -d' ' -f3)"
echo "[✓] Poetry ${POETRY_VER:-}"

# --------------------------------------------------
# 3. Start dependencies (PostgreSQL + Qdrant) with Docker
# --------------------------------------------------
if command -v docker &>/dev/null && command -v docker compose &>/dev/null; then
    echo ""
    echo "==> Starting PostgreSQL and Qdrant..."
    cd "$APP_DIR"
    docker compose up -d ai_agent_db ai_agent_qdrant
    echo "[✓] Dependencies are running"
else
    echo ""
    echo "[!] Docker not found — skipping dependency startup."
    echo "    Make sure PostgreSQL (pgvector) and Qdrant are running manually."
fi

# --------------------------------------------------
# 4. Install project dependencies
# --------------------------------------------------
echo ""
echo "==> Installing project dependencies..."
poetry install --no-interaction --no-ansi --quiet

# --------------------------------------------------
# 5. Run the application
# --------------------------------------------------
echo ""
echo "==> Starting Agent App..."
echo "    App:       http://localhost:2424"
echo "    API Docs:  http://localhost:2424/api-doc"
echo ""
cd "$APP_DIR"
PYTHONPATH="$APP_DIR/src" poetry run python src/agent_app/main.py
