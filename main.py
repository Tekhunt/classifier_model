"""
main.py - Main FastAPI Application with Authentication
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from src.routers import agentic_bot, auth  # Add auth import
from src.models import get_db_context, db_manager
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
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # 1. Load or train sentiment model
    model_loaded = sentiment_analyzer.load_model()
    loaded = sentiment_analyzer.load_model()
    print("Model loaded:", loaded)
    if not model_loaded:
        logger.info("No saved model found — training new model...")
        try:
            sentiment_analyzer.train_model()
            logger.info("Model trained successfully.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")

    # 2. Ensure admin user exists
    with get_db_context() as db:
        from src.models import User, DatabaseManager
        user_count = db.query(User).count()

        if user_count == 0:
            logger.info("Creating default admin user...")
            try:
                admin_user = DatabaseManager.create_user(
                    db=db,
                    email="admin@example.com",
                    username="admin",
                    password="admin123",
                    full_name="System Administrator"
                )
                logger.info(f"Admin user created: {admin_user.username}")
            except Exception as e:
                logger.error(f"Error creating admin user: {e}")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Advanced sentiment analysis API with authentication",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(system.router)  # Public endpoints
app.include_router(auth.router)    # Authentication endpoints (NEW)
app.include_router(analysis.router)  # Protected endpoints
app.include_router(reviews.router)   # Protected endpoints
app.include_router(statistics.router)  # Protected endpoints
app.include_router(agentic_bot.router)  # Protected endpoints

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=8000,
        reload=settings.reload,
        log_level=settings.log_level
    )
