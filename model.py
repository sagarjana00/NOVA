# models.py
from app import db
from datetime import datetime

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), index=True)           # socket session
    # user_id = db.Column(db.Integer, db.ForeignKey('users.id'))  # ← if you add auth later
    role = db.Column(db.String(20), nullable=False)             # user / assistant
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ChatMessage {self.role}: {self.content[:40]}...>"