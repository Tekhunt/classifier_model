# Sentiment Analysis Backend

## Overview

This is the backend API for the Sentiment Analysis Dashboard project. Built with FastAPI, it provides endpoints for analyzing customer reviews (single or batch) using a pre-trained Hugging Face Transformers model for sentiment classification (positive, negative, neutral). It computes basic data analysis stats and returns results in JSON format, ready for consumption by a React frontend.

Key goals:
- Fast, async API handling.
- Efficient sentiment scoring with ML.
- Support for CSV/TXT file uploads for batch processing.
- CORS enabled for local React development.

## Features

- **Single Review Analysis**: Submit a text review and get instant sentiment label, confidence score, and derived category.
- **Batch Analysis**: Upload CSV (with 'review' column) or TXT files for bulk processing.
- **Data Analysis**: Aggregated stats like total reviews, sentiment distribution counts, and average confidence.
- **Interactive Docs**: Auto-generated Swagger UI at `/docs`.
- **Extensible**: Easy to add DB persistence (e.g., SQLite) or advanced analytics.

## Tech Stack

- **Framework**: FastAPI (async Python web framework)
- **ML/NLP**: Hugging Face `transformers` pipeline (default: distilbert-base-uncased-finetuned-sst-2-english)
- **Data Handling**: Pandas for batch processing
- **Other**: Pydantic for models, Uvicorn for serving

## Prerequisites

- Python 3.8+
- pip (package manager)

## Installation

1. Clone the repo (or create a new dir for this backend):
   ```
   mkdir sentiment-backend
   cd sentiment-backend
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Save the provided `main.py` in the root directory.

## Usage

### Running the Server

Start the development server:
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Access the API at `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs` (Swagger UI) or `http://localhost:8000/redoc` (ReDoc).

### API Endpoints

| Method | Endpoint          | Description                          | Request Body/Example                  | Response Example |
|--------|-------------------|--------------------------------------|---------------------------------------|------------------|
| POST   | `/analyze`       | Analyze a single review.            | `{"text": "I love this product!"}`   | `{"text": "...", "sentiment": "positive", "confidence": 0.95, "label": "POSITIVE"}` |
| POST   | `/analyze_batch` | Analyze batch from file upload.     | Multipart form: `file` (CSV/TXT)     | `{"results": [...], "analysis": {"total_reviews": 100, "sentiment_counts": {"positive": 60, ...}, "avg_confidence": 0.82}}` |
| GET    | `/stats`         | Get aggregated stats (placeholder). | N/A                                  | `{"message": "Stats endpoint ready"}` |

- **Sentiment Logic**: 
  - Uses pipeline output: If label is POSITIVE, confidence = score; else, 1 - score.
  - Derived category: positive (>0.5), negative (<0.3), neutral (0.3-0.5).
- **File Uploads**: CSV expects a 'review' column; TXT is line-delimited.

### Testing Endpoints

Use curl or Postman:

- Single analysis:
  ```
  curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"text": "This is terrible."}'
  ```

- Batch (with a sample CSV file):
  ```
  curl -X POST "http://localhost:8000/analyze_batch" -F "file=@reviews.csv"
  ```

## Development

- **Model Swap**: To use VADER (rule-based, lighter): Install `vaderSentiment` and replace the pipeline with `from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer`.
- **Add Database**: You can integrate SQLAlchemy + SQLite for storing results:
  1. `pip install sqlalchemy alembic`
  2. Define models, add CRUD endpoints.
- **Error Handling**: Currently basic; add try-catch for model failures or invalid files.
- **Performance**: For large batches, consider async processing with `asyncio.gather`.

## Frontend Integration

- Set `API_BASE = 'http://localhost:8000'` in React's axios calls.
- CORS is pre-configured for `http://localhost:3000`.
- Frontend consumes `/analyze` for single, `/analyze_batch` for uploads, and renders charts from analysis JSON.

## Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add amazing feature'`).
4. Push to branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## Contact

For questions, open an issue or reach out via the main project repo.

---

*Built with ❤️ for sentiment-powered insights.