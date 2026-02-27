from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker

"""
docker run -d --name chatbot -e POSTGRES_USER=chatbot -e POSTGRES_PASSWORD=12345678 -e POSTGRES_DB=chatbot -p 5432:5432 -v chatbot:/var/lib/postgresql/data postgres:18.1-alpine3.23

use Alembic to migrate the sqlalchemy database
"""
POSTGRESQL = 'postgresql+asyncpg//user:password@localhost/dbname'
SQL_LITE = 'sqlite+aiosqlite:///./test.db'


engine = create_async_engine(SQL_LITE, echo=False)
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass
