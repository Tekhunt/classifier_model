"""
routers/reviews.py - Review Management Endpoints
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional, List
import json

from src.models import ReviewsListResponse, ReviewResponse, SentimentProbabilities, MessageResponse
from src.models import get_db, db_manager
from config import settings

router = APIRouter(prefix="/api", tags=["Reviews"])

@router.get("/reviews", response_model=ReviewsListResponse)
async def get_reviews(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(
        settings.default_page_size, 
        ge=1, 
        le=settings.max_page_size,
        description="Items per page"
    ),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of analyzed reviews
    
    - **page**: Page number (default: 1)
    - **per_page**: Items per page (default: 10, max: 100)
    - **sentiment**: Optional filter by sentiment (positive/negative/neutral)
    """
    try:
        # Validate sentiment value if provided
        if sentiment:
            valid_sentiments = ['positive', 'negative', 'neutral']
            if sentiment.lower() not in valid_sentiments:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid sentiment. Must be one of: {', '.join(valid_sentiments)}"
                )
            sentiment = sentiment.lower()
        
        # Get reviews from database
        result = db_manager.get_reviews(
            db=db, 
            page=page, 
            per_page=per_page, 
            sentiment=sentiment
        )
        
        # Convert to response model
        review_responses = []
        for r in result['reviews']:
            probabilities = r['probabilities']
            if isinstance(probabilities, str):
                probabilities = json.loads(probabilities)
            
            review_responses.append(ReviewResponse(
                id=r['id'],
                text=r['text'],
                sentiment=r['sentiment'],
                confidence=r['confidence'],
                probabilities=SentimentProbabilities(**probabilities),
                timestamp=r['timestamp']
            ))
        
        return ReviewsListResponse(
            reviews=review_responses,
            total=result['total'],
            page=result['page'],
            per_page=result['per_page'],
            total_pages=result['total_pages']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching reviews: {str(e)}")

@router.get("/reviews/{review_id}", response_model=ReviewResponse)
async def get_review_by_id(
    review_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific review by ID
    
    - **review_id**: Unique review identifier
    """
    review = db_manager.get_review_by_id(db, review_id)
    
    if not review:
        raise HTTPException(status_code=404, detail=f"Review with ID {review_id} not found")
    
    review_dict = review.to_dict()
    probabilities = review_dict['probabilities']
    if isinstance(probabilities, str):
        probabilities = json.loads(probabilities)
    
    return ReviewResponse(
        id=review.id,
        text=review.text,
        sentiment=review.sentiment,
        confidence=review.confidence,
        probabilities=SentimentProbabilities(**probabilities),
        timestamp=review.timestamp
    )

@router.get("/reviews/search", response_model=List[ReviewResponse])
async def search_reviews(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    db: Session = Depends(get_db)
):
    """
    Search reviews by text content
    
    - **q**: Search query
    - **limit**: Maximum number of results (default: 10, max: 100)
    """
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")
    
    results = db_manager.search_reviews(db, q, limit)
    
    review_responses = []
    for r in results:
        review_dict = r.to_dict()
        probabilities = review_dict['probabilities']
        if isinstance(probabilities, str):
            probabilities = json.loads(probabilities)
        
        review_responses.append(ReviewResponse(
            id=r.id,
            text=r.text,
            sentiment=r.sentiment,
            confidence=r.confidence,
            probabilities=SentimentProbabilities(**probabilities),
            timestamp=r.timestamp
        ))
    
    return review_responses

@router.get("/reviews/recent", response_model=List[ReviewResponse])
async def get_recent_reviews(
    limit: int = Query(10, ge=1, le=50, description="Number of reviews"),
    db: Session = Depends(get_db)
):
    """
    Get most recent reviews
    
    - **limit**: Number of reviews to return (default: 10, max: 50)
    """
    reviews = db_manager.get_recent_reviews(db, limit)
    
    review_responses = []
    for r in reviews:
        review_dict = r.to_dict()
        probabilities = review_dict['probabilities']
        if isinstance(probabilities, str):
            probabilities = json.loads(probabilities)
        
        review_responses.append(ReviewResponse(
            id=r.id,
            text=r.text,
            sentiment=r.sentiment,
            confidence=r.confidence,
            probabilities=SentimentProbabilities(**probabilities),
            timestamp=r.timestamp
        ))
    
    return review_responses

@router.delete("/clear", response_model=MessageResponse)
async def clear_all_reviews(db: Session = Depends(get_db)):
    """
    Clear all reviews from the database
    
    **Warning**: This action cannot be undone!
    """
    count = db_manager.clear_all(db)
    return MessageResponse(
        message=f"Successfully cleared {count} reviews",
        status="success"
    )
