# Agent App — AI-Powered Natural Language to SQL Query Builder

An intelligent **Text-to-SQL** application that converts natural language questions into executable SQL queries using **LangGraph** state machines, **LLM** reasoning, and **vector search** over database schemas.

Built with a **Clean Architecture** philosophy, the system understands your database schema, retrieves relevant tables semantically, validates sufficiency, plans, generates, validates, and executes SQL — all in a self-correcting pipeline.

---

## Features

- **Natural Language to SQL** — Ask questions in plain English; get executable SQL.
- **Semantic Schema Retrieval** — Uses Qdrant vector DB + embeddings (`BAAI/bge-large-en-v1.5`) to find relevant tables and columns.
- **Multi-Dialect Support** — PostgreSQL, MySQL, MSSQL / SQL Server.
- **Self-Correcting Pipeline** — LangGraph state machine with retry loops:
  - Intent extraction → schema retrieval → schema validation (LLM)
  - SQL generation → static validation → database dry-run (`EXPLAIN`)
  - Execution → result formatting
- **FK-Graph Expansion** — Automatically discovers foreign-key-related tables.
- **Scalar API Documentation** — Interactive OpenAPI docs at `/api-doc`.
- **Docker** — PostgreSQL (pgvector) and Qdrant run in containers; the app runs natively.
- **Poetry** — Modern Python dependency management.

---

## Prerequisites

- **Python** `>=3.10, <3.13`
- **Poetry** `>=2.0` ([install guide](https://python-poetry.org/docs/#installation))
- **Docker** & **Docker Compose** (for PostgreSQL + Qdrant)

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd agent-app
```

### 2. Start dependencies (PostgreSQL + Qdrant)

```bash
docker compose up -d ai_agent_db ai_agent_qdrant
```

Or via Make:

```bash
make deps
```

This starts:
- **PostgreSQL** with pgvector on `:5432`
- **Qdrant** vector database on `:6333` (gRPC) / `:6334` (HTTP)

### 3. Run the application

```bash
# Install dependencies (first time only)
poetry install

# Run
poetry run python src/agent_app/main.py
```

### One-command setup (starts deps + runs app)

```bash
./run.sh        # Unix / Mac
```

```batch
run.bat         # Windows
```

---

## Configuration

All configuration lives in **`config.yaml`** at the project root:

```yaml
app:
  debug: false
  log_level: DEBUG
  http_server:
    host: "0.0.0.0"
    port: 2424

database:
  master:
    driver: "postgresql+psycopg2"
    host: "localhost"
    port: 5432
    username: postgres
    password: postgres
    database: ai_agent_app
    pool:
      size: 20
      max_overflow: 40
      timeout: 30
      recycle: 1800
      pre_ping: true

qdrant:
  host: localhost
  port: 6333

huggingface:
  token:   # Optional: for gated embedding models
```

| Key | Description |
|-----|-------------|
| `app.http_server.host` / `port` | App listen address |
| `database.master` | Main PostgreSQL connection (used to store connection metadata and execute queries) |
| `database.slaves` | Optional read-replicas |
| `qdrant.host` / `port` | Qdrant vector DB connection |
| `huggingface.token` | HuggingFace token for gated models (optional) |

---

## How to Use

### Step 1: Sync Database Schema

Before querying, you need to synchronize your database schema into the vector store.

```bash
POST /v1/database-schema/schema-sync
```

**Request Body:**

```json
{
  "database_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "db_name": "your_database",
  "username": "postgres",
  "password": "postgres"
}
```

> **Note:** If `id` is omitted, a new entry is created. If `id` is provided, the existing schema is updated.

This endpoint:
1. Connects to your database and introspects its schema (tables, columns, foreign keys).
2. Generates vector embeddings for each table schema.
3. Stores embeddings in Qdrant for semantic retrieval.

---

### Step 2: Build a Query

Once your schema is synced, you can ask questions in natural language.

```bash
POST /v1/database-schema/build-query
```

**Request Body:**

```json
{
  "database_id": "<id-from-schema-sync>",
  "user_input": "Show me all customers who placed orders in the last 30 days",
  "query_builder_llm": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4o"
  }
}
```

**Response:**

```json
{
  "sample_data": "[{\"customer_id\": 1, \"name\": \"John\", ...}]",
  "generated_query": "SELECT c.* FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.created_at >= NOW() - INTERVAL '30 days'"
}
```

The builder pipeline:
1. **Extracts intent** — converts your question into semantic search keywords.
2. **Retrieves schemas** — finds relevant tables from the vector store + FK expansion.
3. **Validates** — LLM checks if the retrieved tables are sufficient (re-retrieves if not, up to 3 attempts).
4. **Enhances** — enriches schemas with DDL and sample rows.
5. **Plans & generates** — LLM produces a step-by-step plan then generates dialect-aware SQL.
6. **Validates** — security checks (no DML/DELETE/DROP) + `EXPLAIN` dry-run.
7. **Executes** — runs the query and returns results.
8. **Formats** — conversational response.

---

### Additional Endpoint

**Search Table Schema**

```bash
GET /v1/database-schema/table-schema?database_id=<id>&query=users&limit=5
```

Returns matching table schemas from the vector store without running the full builder.

---

### API Documentation

Interactive Swagger-style docs are available via **Scalar**:

- **Scalar UI:** [`http://localhost:2424/api-doc`](http://localhost:2424/api-doc)
- **OpenAPI JSON:** [`http://localhost:2424/openapi.json`](http://localhost:2424/openapi.json)

---

## Project Structure

```
src/
  agent_app/
    main.py                          # Application entry point
    bootstrap.py                     # Lifecycle manager
    container/
      dependency_injection.py        # DI wiring (singletons, factories)
    agents/
      sql_builder/                   # LangGraph Text-to-SQL pipeline
      response_struct/               # Structured response extraction
    application/
      usecase/database_schema/       # Business logic
      dto/                           # Data transfer objects
    domain/
      entities/database_schema.py    # Domain models (SQLAlchemy)
    infrastructure/
      repository/                    # DB + vector store implementations
    presentation/
      delivery/http/handlers/        # FastAPI route handlers
      middleware/                    # Timeout middleware
    shared/
      config/                        # Configuration loading
      database/connection/           # SQLAlchemy + Qdrant connections
      database/helper/               # Database helper utilities
      embedding/                     # Sentence-transformers embeddings
      logging/                       # Structured JSON logging
```

---

## Makefile Commands

```bash
make deps       # Start PostgreSQL + Qdrant with Docker
make run        # Run the app locally with Poetry
make install    # Install Python dependencies
make lint       # Run ruff linter
make clean      # Stop containers and remove venv
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | FastAPI |
| **AI Orchestration** | LangGraph |
| **LLM Providers** | OpenAI, Groq, Google Gemini |
| **Vector DB** | Qdrant |
| **Embeddings** | sentence-transformers (`BAAI/bge-large-en-v1.5`) |
| **Database** | PostgreSQL (pgvector) |
| **DI** | dependency-injector |
| **API Docs** | Scalar |
| **Package Manager** | Poetry |
| **Containerization** | Docker / Docker Compose (deps only) |
