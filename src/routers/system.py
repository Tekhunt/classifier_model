"""
routers/system.py - System / Health Check Endpoints
"""

from fastapi import APIRouter
from config import settings
from src.ml_models import sentiment_analyzer

router = APIRouter(prefix="/api", tags=["System"])

@router.get("/", summary="Root endpoint")
async def root():
    """Root endpoint"""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "message": "Welcome to the Sentiment Analysis API!"
    }

@router.get("/health", summary="Health check")
async def health_check():
    """Simple health check endpoint"""
    model_status = sentiment_analyzer.get_model_info()
    return {
        "status": "ok",
        "model_loaded": model_status['is_trained'],
        "model_type": model_status['model_type'],
        "vectorizer_type": model_status['vectorizer_type']
    }

@router.get("/status", summary="System status")
async def system_status():
    """Return system status info"""
    model_info = sentiment_analyzer.get_model_info()
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model_trained": model_info['is_trained'],
        "training_metrics": model_info.get('metrics', {}),
        "model_path": model_info.get('model_path'),
        "vectorizer_path": model_info.get('vectorizer_path')
    }
