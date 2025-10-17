"""
database.py - SQLite Database with SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import Field
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Generator
import json
import os

from config import settings

# Database configuration
DATABASE_URL = settings.database_url or "sqlite:///./sentiment_analysis.db"
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class Review(Base):
    """SQLAlchemy model for reviews"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(Text, nullable=False)  # Store JSON as text for SQLite
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'probabilities': json.loads(self.probabilities) if isinstance(self.probabilities, str) else self.probabilities,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

# Create tables
Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    """Context manager for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    """Manager class for database operations"""
    
    @staticmethod
    def add_review(db: Session, text: str, sentiment: str, 
                   confidence: float, probabilities: Dict[str, float]) -> Review:
        """
        Add a new review to the database
        """
        review = Review(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=json.dumps(probabilities),
            timestamp=datetime.utcnow()
        )
        db.add(review)
        db.commit()
        db.refresh(review)
        return review
    
    @staticmethod
    def get_reviews(db: Session, page: int = 1, per_page: int = 10, 
                    sentiment: Optional[str] = None) -> Dict:
        """
        Get paginated reviews
        """
        query = db.query(Review)
        
        if sentiment:
            query = query.filter(Review.sentiment == sentiment)
        
        # Get total count
        total = query.count()
        
        # Calculate pagination
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page
        
        # Get paginated results
        reviews = query.order_by(Review.timestamp.desc()).offset(offset).limit(per_page).all()
        
        return {
            'reviews': [r.to_dict() for r in reviews],
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages
        }
    
    @staticmethod
    def get_review_by_id(db: Session, review_id: int) -> Optional[Review]:
        """
        Get a specific review by ID
        """
        return db.query(Review).filter(Review.id == review_id).first()
    
    @staticmethod
    def search_reviews(db: Session, query: str, limit: int = 10) -> List[Review]:
        """
        Search reviews by text content
        """
        return db.query(Review).filter(
            Review.text.contains(query)
        ).limit(limit).all()
    
    @staticmethod
    def get_statistics(db: Session) -> Dict:
        """
        Get sentiment distribution statistics
        """
        total = db.query(Review).count()
        positive = db.query(Review).filter(Review.sentiment == 'positive').count()
        negative = db.query(Review).filter(Review.sentiment == 'negative').count()
        neutral = db.query(Review).filter(Review.sentiment == 'neutral').count()
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'total': total
        }
    
    @staticmethod
    def get_recent_reviews(db: Session, limit: int = 10) -> List[Review]:
        """
        Get most recent reviews
        """
        return db.query(Review).order_by(
            Review.timestamp.desc()
        ).limit(limit).all()
    
    @staticmethod
    def clear_all(db: Session) -> int:
        """
        Clear all reviews from database
        """
        count = db.query(Review).count()
        db.query(Review).delete()
        db.commit()
        return count
    
    @staticmethod
    def get_reviews_by_sentiment(db: Session, sentiment: str) -> List[Review]:
        """
        Get all reviews with specific sentiment
        """
        return db.query(Review).filter(Review.sentiment == sentiment).all()

# Create global database manager
db_manager = DatabaseManager()


"""
models.py - Database and Pydantic Models for Sentiment Analysis
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Dict, Generator
import json
import os

from pydantic import BaseModel
from config import settings

# -------------------
# Database Setup
# -------------------

DATABASE_URL = settings.database_url or "sqlite:///./sentiment_analysis.db"
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'probabilities': json.loads(self.probabilities) if isinstance(self.probabilities, str) else self.probabilities,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------
# Database Manager
# -------------------

class DatabaseManager:
    @staticmethod
    def add_review(db: Session, text: str, sentiment: str, confidence: float, probabilities: Dict[str, float]) -> Review:
        review = Review(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities=json.dumps(probabilities),
            timestamp=datetime.utcnow()
        )
        db.add(review)
        db.commit()
        db.refresh(review)
        return review

    @staticmethod
    def get_reviews(db: Session, page: int = 1, per_page: int = 10, sentiment: Optional[str] = None) -> Dict:
        query = db.query(Review)
        if sentiment:
            query = query.filter(Review.sentiment == sentiment)
        total = query.count()
        total_pages = (total + per_page - 1) // per_page if total > 0 else 0
        offset = (page - 1) * per_page
        reviews = query.order_by(Review.timestamp.desc()).offset(offset).limit(per_page).all()
        return {
            'reviews': [r.to_dict() for r in reviews],
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages
        }

    @staticmethod
    def get_review_by_id(db: Session, review_id: int) -> Optional[Review]:
        return db.query(Review).filter(Review.id == review_id).first()

    @staticmethod
    def search_reviews(db: Session, query: str, limit: int = 10) -> List[Review]:
        return db.query(Review).filter(Review.text.contains(query)).limit(limit).all()

    @staticmethod
    def get_statistics(db: Session) -> Dict:
        total = db.query(Review).count()
        positive = db.query(Review).filter(Review.sentiment == 'positive').count()
        negative = db.query(Review).filter(Review.sentiment == 'negative').count()
        neutral = db.query(Review).filter(Review.sentiment == 'neutral').count()
        return {'positive': positive, 'negative': negative, 'neutral': neutral, 'total': total}

    @staticmethod
    def get_recent_reviews(db: Session, limit: int = 10) -> List[Review]:
        return db.query(Review).order_by(Review.timestamp.desc()).limit(limit).all()

    @staticmethod
    def clear_all(db: Session) -> int:
        count = db.query(Review).count()
        db.query(Review).delete()
        db.commit()
        return count

db_manager = DatabaseManager()

# -------------------
# Pydantic Models / Schemas
# -------------------

class SentimentProbabilities(BaseModel):
    positive: float
    negative: float
    neutral: float

class ReviewRequest(BaseModel):
    text: str

class BatchReviewRequest(BaseModel):
    reviews: List[str]

class ReviewResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    confidence: float
    probabilities: SentimentProbabilities
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # timestamp: Optional[datetime]

class BatchAnalysisResponse(BaseModel):
    results: List[ReviewResponse]
    count: int
    failed: int

class ReviewsListResponse(BaseModel):
    reviews: List[ReviewResponse]
    total: int
    page: int
    per_page: int
    total_pages: int

class MessageResponse(BaseModel):
    message: str
    status: str

class StatisticsResponse(BaseModel):
    positive: int
    negative: int
    neutral: int
    total: int
