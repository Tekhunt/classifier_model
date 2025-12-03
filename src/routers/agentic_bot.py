
"""
FastAPI Application for Agentic RAG System
Provides REST API endpoints for context-aware chat
"""
from fastapi import APIRouter, FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime
import uuid

# Import the chat function from your chatbot module
from .chatbot import chat

# ==================== PYDANTIC MODELS ====================
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., description="User's question", min_length=1, max_length=5000)
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation thread ID for maintaining context across messages"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are LLM Guardrails?",
                "thread_id": "user_123_session_456"
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    response: str
    thread_id: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    version: str


# ==================== ROUTER ====================
router = APIRouter(prefix="/api", tags=["Agentic RAG"])


# ==================== ENDPOINTS ====================

@router.get("/rooty", response_model=HealthResponse)
async def rootly():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@router.get("/healther", response_model=HealthResponse)
async def health_checker():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy ok",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - processes user questions with context retrieval
    
    ## Description
    This endpoint processes user questions about LLM Guardrails using the context_store knowledge base.
    
    ## Parameters
    - **question**: The user's question (required)
    - **thread_id**: Conversation thread ID for memory (auto-generated if not provided)
    
    ## Returns
    - **success**: Whether the request was successful
    - **response**: The AI's response
    - **thread_id**: The conversation thread ID
    - **timestamp**: When the response was generated
    
    ## Example
    ```json
    {
        "question": "What are LLM Guardrails?",
        "thread_id": "user_123"
    }
    ```
    """
    
    try:
        # Generate thread_id if not provided
        thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:8]}"
        
        # Call the chat system (simplified - no user_source needed)
        response = await chat(
            question=request.question,
            thread_id=thread_id
        )
        
        return ChatResponse(
            success=True,
            response=response,
            thread_id=thread_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "question_length": len(request.question),
                "response_length": len(response)
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint (placeholder for future implementation)
    """
    raise HTTPException(
        status_code=501,
        detail="Streaming endpoint not yet implemented"
    )


# ==================== MAIN ====================
if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:router",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
