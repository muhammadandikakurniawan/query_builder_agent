from dataclasses import dataclass, field
from typing import List, Dict
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL
from agent_app.shared.database.connection.model import DRIVER_MAP, DatabaseConnectionConfig, PoolConfig, RetryConfig

def create_connection(config: DatabaseConnectionConfig) -> Engine:
    db_uri = ""
    connect_args = {}
    db_uri = URL.create(
        drivername=config.driver,
        username=config.username,
        password=config.password,
        host=config.host,
        port=config.port,
        database=config.database,
    )

    if config.ssl and config.ssl.enabled:
        connect_args["sslmode"] = config.ssl.mode
        if config.ssl.root_cert:
            connect_args["sslrootcert"] = config.ssl.root_cert

    if config.pool is None:
        config.pool = PoolConfig()

    if config.retry is None:
        config.retry = RetryConfig()
    
    engine = create_engine(
        db_uri,
        pool_size=config.pool.size,
        max_overflow=config.pool.max_overflow,
        pool_timeout=config.pool.timeout,
        pool_recycle=config.pool.recycle,
        pool_pre_ping=config.pool.pre_ping,
        connect_args=connect_args,
        echo=False,
        future=True,
    )

    return engine

def check_database(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False