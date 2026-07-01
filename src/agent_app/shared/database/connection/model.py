from pydantic import BaseModel

DRIVER_MAP = {
    "postgresql": "postgresql+psycopg2",
    "postgres": "postgresql+psycopg2",
    "mysql": "mysql+pymysql",
    "mssql": "mssql+pyodbc",
    "sqlserver": "mssql+pyodbc"
}


class PoolConfig(BaseModel):
    size: int = 20
    max_overflow: int = 40
    timeout: int = 30
    recycle: int = 1800
    pre_ping: bool = True


class SSLConfig(BaseModel):
    enabled: bool = False
    mode: str = "verify-full"
    root_cert: str | None = None


class RetryConfig(BaseModel):
    max_attempts: int = 5
    backoff_seconds: int = 2


class DatabaseConnectionConfig(BaseModel):
    driver: str
    host: str
    port: int
    database: str
    username: str
    password: str
    pool: PoolConfig | None= None
    ssl: SSLConfig | None= None
    retry: RetryConfig | None = None
    echo: bool = False



