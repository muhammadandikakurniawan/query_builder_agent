# main.py

import asyncio
from agent_app.bootstrap import Application, new_database_migrator
from agent_app.container.dependency_injection import Container
from agent_app.presentation.delivery.http.server import HTTPServer
from agent_app.shared.config.model import load_config
from agent_app.shared.logging.logger import setup_logging


async def main():

    config = load_config()
    setup_logging(config.app.log_level)
    container = Container(config=config)
    app = Application()

    app.register(
        HTTPServer(
            container=container,
            host=config.app.http_server.host,
            port=config.app.http_server.port,
        )
    )

    app.register(new_database_migrator(db_helper=container.db_manager().get_db_master()))

    await app.run()

if __name__ == "__main__":
    asyncio.run(main())