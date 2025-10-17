"""
main.py - Main FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from src.models import get_db_context, db_manager
# from src.routers.analysis 
from src.routers import analysis, reviews, statistics, system
from src.utils import load_sample_reviews
from src.ml_models import sentiment_analyzer, SentimentAnalyzer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Load or train model
    if not sentiment_analyzer.load_model():
        logger.info("No saved model found, training new model...")
        result = sentiment_analyzer.train_model()
        if result['status'] == 'success':
            logger.info(f"Model trained successfully - Accuracy: {result['metrics']['accuracy']:.4f}")
        else:
            logger.warning("Model training failed, API will train on first request")
    
    # Load sample reviews into database if empty
    with get_db_context() as db:
        stats = db_manager.get_statistics(db)
        if stats['total'] == 0:
            logger.info("Loading sample reviews...")
            sample_reviews = load_sample_reviews()[:5]  # Load first 5 samples
            for text in sample_reviews:
                try:
                    prediction = sentiment_analyzer.predict(text)
                    db_manager.add_review(
                        db=db,
                        text=text,
                        sentiment=prediction['sentiment'],
                        confidence=prediction['confidence'],
                        probabilities=prediction['probabilities']
                    )
                except Exception as e:
                    logger.error(f"Error adding sample review: {e}")
            logger.info(f"Loaded {len(sample_reviews)} sample reviews")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Advanced sentiment analysis API for product reviews using machine learning",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(system.router)
app.include_router(analysis.router)
app.include_router(reviews.router)
app.include_router(statistics.router)

# Root endpoint is in system router

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level
    )