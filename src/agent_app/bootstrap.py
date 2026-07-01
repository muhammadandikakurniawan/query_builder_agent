# bootstrap/application.py

import asyncio
import signal

from agent_app.domain.entities.database_schema import DatabaseConnectionEntity
from agent_app.shared.database.helper.database_helper import DatabaseHelper
from agent_app.shared.model.lifecycle import Lifecycle



class Application:

    def __init__(self):
        self._components: list[Lifecycle] = []
        self._shutdown_event = asyncio.Event()

    def register(self, component: Lifecycle):
        self._components.append(component )

    async def start(self):
        for component in self._components:
            await component.start()

    async def stop(self):
        for component in reversed(self._components):
            await component.stop()

    async def run(self):

        loop = asyncio.get_running_loop()

        loop.add_signal_handler(
            signal.SIGINT,
            self._shutdown_event.set,
        )

        loop.add_signal_handler(
            signal.SIGTERM,
            self._shutdown_event.set,
        )

        await self.start()

        await self._shutdown_event.wait()

        await self.stop()



class DatabaseMigrator(Lifecycle):

    def __init__(self, db_helper: DatabaseHelper):
        self._db_helper = db_helper

    async def start(self) -> None:
        self._db_helper.migrate(
            DatabaseConnectionEntity
        )
        return

    async def stop(self) -> None:
        return None

def new_database_migrator(db_helper: DatabaseHelper) -> DatabaseMigrator:
    return DatabaseMigrator(db_helper = db_helper)