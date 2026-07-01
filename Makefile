.PHONY: install deps run lint clean

install:
	poetry install --no-interaction --no-ansi

deps:
	docker compose up -d ai_agent_db ai_agent_qdrant

run:
	PYTHONPATH=src poetry run python src/agent_app/main.py

lint:
	poetry run ruff check src/

clean:
	-poetry env remove --all 2>/dev/null
	docker compose down -v 2>/dev/null