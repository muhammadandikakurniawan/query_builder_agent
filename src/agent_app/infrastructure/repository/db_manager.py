from typing import Any, Union

from sqlalchemy import Inspector, Result, create_engine, inspect, text
from sqlalchemy.engine import Engine, URL
import random

from sqlalchemy.orm import Session

from agent_app.shared.config.model import DatabaseConfig
from agent_app.shared.database.helper.database_helper import DatabaseHelper

class DatabaseManager:
    def __init__(self, database: Union[DatabaseConfig, Session]):
        self._db_master : DatabaseHelper | None = None
        self._db_slaves: list[DatabaseHelper] = []
        self._session: Session | None = None
        if isinstance(database, DatabaseConfig):
            self._db_master = DatabaseHelper(config=database.master)
            self._db_slaves = [
                DatabaseHelper(config=cfg) for cfg in database.slaves
            ] or [self._db_master]
        elif isinstance(database, Session):
            self._session = database

    def execute_master(self, sql: str, params: Union[dict[str, Any], tuple[Any, ...], list[Any], None] = None) -> Result[Any]:
        if self._session:
            return self._session.execute(text(sql), params)
        
        return self._db_master.execute(sql, params)

    def execute_slave(self, sql: str, params: Union[dict[str, Any], tuple[Any, ...], list[Any], None] = None) -> Result[Any]:
        if self._session:
            return self._session.execute(text(sql), params)

        slave = random.choice(self._db_slaves)
        return slave.execute(sql, params)

    def inspect(self) -> Inspector:
        if self._session:
            return inspect(self._session)
        
        return self._db_master.inspect()

    def get_db_master(self) -> DatabaseHelper:
        return self._db_master