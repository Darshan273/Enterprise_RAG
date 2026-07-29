from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from app.database.connection import Base

class ConversationHistory(Base):
    __tablename__ = "conversational_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    user_message = Column(String, nullable=False)
    ai_message = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
