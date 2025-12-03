"""
models.py - Database Models with User Authentication
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Generator
import json
import os
from passlib.context import CryptContext
from jose import JWTError, jwt

from pydantic import BaseModel, EmailStr, Field
from config import settings

# -------------------
# Security Configuration
# -------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Settings (add these to your config.py or use directly)
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# -------------------
# Database Setup
# -------------------

DATABASE_URL = settings.database_url or "sqlite:///./sentiment_analysis.db"
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# -------------------
# User Model
# -------------------

class User(Base):
    """SQLAlchemy model for users"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with reviews
    # reviews = relationship("Review", back_populates="user")
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# -------------------
# Review Model (Updated)
# -------------------

class Review(Base):
    """SQLAlchemy model for reviews"""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    # user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    sentiment = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    probabilities = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship with user
    # user = relationship("User", back_populates="reviews")

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            # 'user_id': self.user_id,
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'probabilities': json.loads(self.probabilities) if isinstance(self.probabilities, str) else self.probabilities,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


# Create tables
Base.metadata.create_all(bind=engine)


# -------------------
# Database Session Management
# -------------------

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
# Password & Token Utilities
# -------------------

class AuthUtils:
    """Authentication utility functions"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict]:
        """Decode and verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None


# -------------------
# Database Manager (Updated)
# -------------------

class DatabaseManager:
    
    # -------- User Operations --------
    
    @staticmethod
    def create_user(db: Session, email: str, username: str, password: str, 
                    full_name: Optional[str] = None) -> User:
        """Create a new user"""
        hashed_password = AuthUtils.get_password_hash(password)
        user = User(
            email=email,
            username=username,
            full_name=full_name,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = DatabaseManager.get_user_by_username(db, username)
        if not user:
            return None
        if not AuthUtils.verify_password(password, user.hashed_password):
            return None
        return user
    
    # -------- Review Operations (Updated) --------
    # user_id: int,
    @staticmethod
    def add_review(db: Session, text: str, sentiment: str, 
                   confidence: float, probabilities: Dict[str, float]) -> Review:
        """Add a new review to the database"""
        review = Review(
            # user_id=user_id,
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
                    sentiment: Optional[str] = None, user_id: Optional[int] = None) -> Dict:
        """Get paginated reviews (optionally filtered by user)"""
        query = db.query(Review)
        
        if sentiment:
            query = query.filter(Review.sentiment == sentiment)
        
        # if user_id:
        #     query = query.filter(Review.user_id == user_id)
        
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
        """Get a specific review by ID"""
        return db.query(Review).filter(Review.id == review_id).first()
    
    @staticmethod
    def search_reviews(db: Session, query: str, limit: int = 10, user_id: Optional[int] = None) -> List[Review]:
        """Search reviews by text content"""
        q = db.query(Review).filter(Review.text.contains(query))
        if user_id:
            q = q.filter(Review.user_id == user_id)
        return q.limit(limit).all()
    
    @staticmethod
    def get_statistics(db: Session, user_id: Optional[int] = None) -> Dict:
        """Get sentiment distribution statistics"""
        query = db.query(Review)
        if user_id:
            query = query.filter(Review.user_id == user_id)
        
        total = query.count()
        positive = query.filter(Review.sentiment == 'positive').count()
        negative = query.filter(Review.sentiment == 'negative').count()
        neutral = query.filter(Review.sentiment == 'neutral').count()
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'total': total
        }
    
    @staticmethod
    def get_recent_reviews(db: Session, limit: int = 10, user_id: Optional[int] = None) -> List[Review]:
        """Get most recent reviews"""
        query = db.query(Review)
        if user_id:
            query = query.filter(Review.user_id == user_id)
        return query.order_by(Review.timestamp.desc()).limit(limit).all()
    
    @staticmethod
    def clear_all(db: Session, user_id: Optional[int] = None) -> int:
        """Clear reviews (optionally only for specific user)"""
        query = db.query(Review)
        if user_id:
            query = query.filter(Review.user_id == user_id)
        count = query.count()
        query.delete()
        db.commit()
        return count


db_manager = DatabaseManager()


# -------------------
# Pydantic Schemas
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
    # user_id: int
    text: str
    sentiment: str
    confidence: float
    probabilities: SentimentProbabilities
    timestamp: datetime = Field(default_factory=datetime.utcnow)

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

# Authentication Schemas

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
