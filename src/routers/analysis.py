"""
routers/analysis.py - Sentiment Analysis Endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List
import logging

from src.models import (
    ReviewRequest, 
    BatchReviewRequest, 
    ReviewResponse, 
    BatchAnalysisResponse,
    SentimentProbabilities
)
from src.ml_models import sentiment_analyzer
from src.models import get_db, db_manager
from src.utils import validate_review_text, clean_review_batch

router = APIRouter(prefix="/api", tags=["Analysis"])
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=ReviewResponse)
async def analyze_sentiment(
    request: ReviewRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment of a single review
    
    - **text**: The review text to analyze (1-10000 characters)
    
    Returns sentiment prediction with confidence score and probability distribution
    """
    # Validate input
    is_valid, error_message = validate_review_text(request.text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)
    
    try:
        # Check if model is trained
        if not sentiment_analyzer.load_model():
            print("Model does not exist or is not trained.")
            # result = sentiment_analyzer.train_model()
            # if result['status'] != 'success':
            #     raise HTTPException(
            #         status_code=503, 
            #         detail="Model training failed. Please try again later."
            #     )
    
        # Predict sentiment
        prediction = sentiment_analyzer.predict(request.text)
        
        # Store in database
        review = db_manager.add_review(
            db=db,
            text=request.text,
            sentiment=prediction['sentiment'],
            confidence=prediction['confidence'],
            probabilities=prediction['probabilities']
        )
        
        # Log analysis in background
        background_tasks.add_task(log_analysis, review.id, review.sentiment, review.confidence)
        
        # Create response
        return ReviewResponse(
            id=review.id,
            text=review.text,
            sentiment=review.sentiment,
            confidence=review.confidence,
            probabilities=SentimentProbabilities(**prediction['probabilities']),
            timestamp=review.timestamp
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/batch_analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_sentiments(
    request: BatchReviewRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze multiple reviews at once
    
    - **reviews**: List of review texts (1-100 items)
    
    Returns analysis results for all reviews with success/failure counts
    """
    # Clean and validate reviews
    cleaned_reviews = clean_review_batch(request.reviews)
    
    if not cleaned_reviews:
        raise HTTPException(status_code=400, detail="No valid reviews provided")
    
    try:
        # Check if model is trained
        if not sentiment_analyzer.is_trained:
            if not sentiment_analyzer.load_model():
                result = sentiment_analyzer.train_model()
                if result['status'] != 'success':
                    raise HTTPException(
                        status_code=503,
                        detail="Model not available. Please try again later."
                    )
        
        results = []
        failed = 0
        
        for review_text in cleaned_reviews:
            try:
                # Validate each review
                is_valid, _ = validate_review_text(review_text)
                if not is_valid:
                    failed += 1
                    continue
                
                # Predict sentiment
                prediction = sentiment_analyzer.predict(review_text)
                
                # Store in database
                review = db_manager.add_review(
                    db=db,
                    text=review_text,
                    sentiment=prediction['sentiment'],
                    confidence=prediction['confidence'],
                    probabilities=prediction['probabilities']
                )
                
                # Add to results
                results.append(ReviewResponse(
                    id=review.id,
                    text=review.text,
                    sentiment=review.sentiment,
                    confidence=review.confidence,
                    probabilities=SentimentProbabilities(**prediction['probabilities']),
                    timestamp=review.timestamp
                ))
                
            except Exception as e:
                logger.error(f"Error processing review: {e}")
                failed += 1
                continue
        
        return BatchAnalysisResponse(
            results=results,
            count=len(results),
            failed=failed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

def log_analysis(review_id: int, sentiment: str, confidence: float):
    """Background task to log analysis results"""
    logger.info(f"Analysis logged - ID: {review_id}, Sentiment: {sentiment}, Confidence: {confidence:.2f}")
    