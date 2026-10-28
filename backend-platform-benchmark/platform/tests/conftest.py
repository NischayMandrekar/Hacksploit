import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def test_database_url():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a dedicated PostgreSQL test database; SQLite is not equivalent for SKIP LOCKED.")
    return url

@pytest.fixture(scope="session")
def engine(test_database_url):
    engine = create_engine(test_database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    yield engine
    engine.dispose()

@pytest.fixture
def db_session(engine):
    from platform_app.models import Base
    Base.metadata.drop_all(engine); Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        yield session
        session.rollback()
