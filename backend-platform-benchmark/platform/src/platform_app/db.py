from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session
from platform_app.config import settings

engine_options = {"pool_pre_ping": True}
# Keep PostgreSQL pool sizing explicit. Lightweight SQLite imports may be used for
# non-database smoke checks, but SQLite is not supported for correctness tests.
if make_url(settings.database_url).get_backend_name() != "sqlite":
    engine_options.update(pool_size=5, max_overflow=10)
engine = create_engine(settings.database_url, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
