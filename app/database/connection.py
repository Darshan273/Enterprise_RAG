from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import settings
import logfire

with logfire.span("Connecting to PostgreSQL"):
    engine = create_engine(settings.POSTGRESQL_URL)
    SessionLocal = sessionmaker(autocommit=False, 
                                autoflush=False, 
                                bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

    
