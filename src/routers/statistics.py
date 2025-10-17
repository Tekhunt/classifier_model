"""
routers/statistics.py - Statistics and Analytics Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict
import json

# from src.models import StatisticsResponse
from src.models import get_db, db_manager, Review, StatisticsResponse

router = APIRouter(prefix="/api", tags=["Statistics"])

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """
    Get sentiment distribution statistics
    
    Returns counts and percentages for each sentiment category
    """
    try:
        stats = db_manager.get_statistics(db)
        
        return StatisticsResponse(
            positive=stats['positive'],
            negative=stats['negative'],
            neutral=stats['neutral'],
            total=stats['total']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")

@router.get("/statistics/detailed")
async def get_detailed_statistics(db: Session = Depends(get_db)):
    """
    Get detailed statistics including averages and trends
    
    Returns comprehensive analytics data
    """
    try:
        # Get basic statistics
        basic_stats = db_manager.get_statistics(db)
        
        # Calculate additional metrics using SQLAlchemy
        if basic_stats['total'] > 0:
            # Average confidence across all reviews
            avg_confidence = db.query(func.avg(Review.confidence)).scalar() or 0
            
            # Average confidence by sentiment
            sentiment_confidence = {}
            for sentiment in ['positive', 'negative', 'neutral']:
                avg_conf = db.query(func.avg(Review.confidence)).filter(
                    Review.sentiment == sentiment
                ).scalar()
                sentiment_confidence[sentiment] = avg_conf or 0
            
            # Average review length
            avg_length = db.query(func.avg(func.length(Review.text))).scalar() or 0
            
            # Get date range of reviews
            oldest_review = db.query(func.min(Review.timestamp)).scalar()
            newest_review = db.query(func.max(Review.timestamp)).scalar()
            
        else:
            avg_confidence = 0
            sentiment_confidence = {'positive': 0, 'negative': 0, 'neutral': 0}
            avg_length = 0
            oldest_review = None
            newest_review = None
        
        return {
            'basic_stats': basic_stats,
            'averages': {
                'confidence': round(float(avg_confidence), 3),
                'review_length': round(float(avg_length), 1)
            },
            'confidence_by_sentiment': {
                k: round(float(v), 3) for k, v in sentiment_confidence.items()
            },
            'percentages': {
                'positive': round((basic_stats['positive'] / max(basic_stats['total'], 1)) * 100, 2),
                'negative': round((basic_stats['negative'] / max(basic_stats['total'], 1)) * 100, 2),
                'neutral': round((basic_stats['neutral'] / max(basic_stats['total'], 1)) * 100, 2)
            },
            'date_range': {
                'oldest': oldest_review.isoformat() if oldest_review else None,
                'newest': newest_review.isoformat() if newest_review else None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating detailed statistics: {str(e)}")
    