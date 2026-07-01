from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Union

from sqlalchemy import Inspector, create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer
from agent_app.shared.database.connection.model import DatabaseConnectionConfig, PoolConfig, RetryConfig

class BaseTable(DeclarativeBase):
    pass

class DatabaseHelper:
    """
    Production-ready SQLAlchemy helper.

    Features:
    - Engine management
    - Session factory
    - Transaction handling
    - Health checks
    - Graceful shutdown
    """

    def __init__(
        self,
        *,
        config: DatabaseConnectionConfig
    ):
        url = URL.create(
            drivername=config.driver,
            username=config.username,
            password=config.password,
            host=config.host,
            port=config.port,
            database=config.database,
        )

        connect_args = {}

        if config.ssl and config.ssl.enabled:
            connect_args["sslmode"] = config.ssl.mode
            if cfg.ssl.root_cert:
                connect_args["sslrootcert"] = config.ssl.root_cert

        if config.pool is None:
            config.pool = PoolConfig()

        if config.retry is None:
            config.retry = RetryConfig()

        self.engine: Engine = create_engine(
            url,
            pool_size=config.pool.size,
            max_overflow=config.pool.max_overflow,
            pool_timeout=config.pool.timeout,
            pool_recycle=config.pool.recycle,
            pool_pre_ping=config.pool.pre_ping,
            connect_args=connect_args,
            echo=config.echo,
            future=True,
        )

        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Transaction-aware session.

        Commits on success.
        Rolls back on exception.
        """
        session = self.session_factory()

        try:
            yield session
            session.commit()

        except Exception:
            session.rollback()
            raise

        finally:
            session.close()

    def get_session(self) -> Session:
        """
        Manual session handling.
        """
        return self.session_factory()

    def inspect(self) -> Inspector:
        return inspect(self.engine)

    def health_check(self) -> bool:
        """
        Verify database connectivity.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def execute(self, sql: str, params: Union[dict[str, Any], tuple[Any, ...], list[Any], None] = None):
        """
        Execute raw SQL.
        """
        with self.session() as session:

            stmt = text(sql)

            # dict → named parameters
            if isinstance(params, dict) or params is None:
                result = session.execute(stmt, params or {})

            # tuple/list → positional parameters
            else:
                result = session.execute(stmt, tuple(params))

            return result

    def dispose(self):
        """
        Close connection pool.
        """
        self.engine.dispose()

    def migrate(self, *models: type[BaseTable]) -> None:
        """
        Create tables for the given models only.
        """
        tables = [model.__table__ for model in models]
        BaseTable.metadata.create_all(self.engine, tables=tables)