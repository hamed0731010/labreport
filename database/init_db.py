"""Initialize missing database tables and PostgreSQL types."""

import asyncio

from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from src.database.connection import async_engine
from src.core.config import DATABASE_URL
from src.database.models import Base


async def initialize_database() -> None:
    database_url = make_url(DATABASE_URL)
    print(
        "Connecting to PostgreSQL "
        f"host={database_url.host} port={database_url.port} "
        f"database={database_url.database}",
        flush=True,
    )
    try:
        async with async_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            tables = await connection.run_sync(
                lambda sync_connection: set(inspect(sync_connection).get_table_names())
            )
            database, schema = (
                await connection.execute(
                    text("SELECT current_database(), current_schema()")
                )
            ).one()

        expected_tables = set(Base.metadata.tables)
        missing_tables = expected_tables - tables
        if missing_tables:
            raise RuntimeError(
                "Schema initialization did not create expected tables: "
                + ", ".join(sorted(missing_tables))
            )

        print(
            f"Database schema ready: database={database}, schema={schema}; "
            f"tables={', '.join(sorted(expected_tables))}"
        )
    finally:
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(initialize_database())
