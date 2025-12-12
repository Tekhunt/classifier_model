
"""
FastAPI Application for Agentic RAG System
Provides REST API endpoints for context-aware chat
"""
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Body, logger
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime
import uuid
from src.models import User
from src.routers.auth import get_current_active_user
from .chatbot import ConversationStore, chat
import logging

logger = logging.getLogger(__name__)
conversation_store = ConversationStore()

# ==================== PYDANTIC MODELS ====================
# class ChatRequest(BaseModel):
#     """Request model for chat endpoint"""
#     question: str = Field(..., description="User's question", min_length=1, max_length=5000)
#     thread_id: Optional[str] = Field(
#         default=None,
#         description="Conversation thread ID for maintaining context across messages"
#     )
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "question": "What are LLM Guardrails?",
#                 "thread_id": "user_123_session_456"
#             }
#         }


# class ChatResponse(BaseModel):
#     """Response model for chat endpoint"""
#     success: bool
#     response: str
#     thread_id: str
#     timestamp: str
#     metadata: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="User's question")
    user_id: Optional[str] = Field(None, description="Optional user ID override")

class ChatResponse(BaseModel):
    success: bool
    response: str
    thread_id: str
    timestamp: str
    metadata: Optional[dict] = None


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


# @router.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     """
#     Main chat endpoint - processes user questions with context retrieval
    
#     ## Description
#     This endpoint processes user questions about LLM Guardrails using the context_store knowledge base.
    
#     ## Parameters
#     - **question**: The user's question (required)
#     - **thread_id**: Conversation thread ID for memory (auto-generated if not provided)
    
#     ## Returns
#     - **success**: Whether the request was successful
#     - **response**: The AI's response
#     - **thread_id**: The conversation thread ID
#     - **timestamp**: When the response was generated
    
#     ## Example
#     ```json
#     {
#         "question": "What are LLM Guardrails?",
#         "thread_id": "user_123"
#     }
#     ```
#     """
    
#     try:
#         # Generate thread_id if not provided
#         # thread_id = request.thread_id or f"thread_{uuid.uuid4().hex[:8]}"
       
#         thread_id = "12345678"  # Temporary hardcode for testing
#         # Call the chat system (simplified - no user_source needed)
#         response = await chat(
#             question=request.question,
#             thread_id=thread_id
#         )
        
#         return ChatResponse(
#             success=True,
#             response=response,
#             thread_id=thread_id,
#             timestamp=datetime.utcnow().isoformat(),
#             metadata={
#                 "question_length": len(request.question),
#                 "response_length": len(response)
#             }
#         )
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "error": "Internal server error",
#                 "message": str(e),
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#         )


# @router.post("/chat/stream")
# async def chat_stream_endpoint(request: ChatRequest):
#     """
#     Streaming chat endpoint (placeholder for future implementation)
#     """
#     raise HTTPException(
#         status_code=501,
#         detail="Streaming endpoint not yet implemented"
#     )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user)  # Add authentication
):
    """
    Main chat endpoint - processes user questions with context retrieval (Protected)
    
    ## Description
    This endpoint processes user questions about LLM TechGadgets using the context_store knowledge base.
    Each user has their own conversation history managed by user_id.
    
    ## Authentication
    Requires valid JWT token in Authorization header.
    
    ## Parameters
    - **question**: The user's question (required)
    - **user_id**: Optional user ID override (uses authenticated user by default)
    
    ## Returns
    - **success**: Whether the request was successful
    - **response**: The AI's response
    - **thread_id**: The conversation thread ID
    - **timestamp**: When the response was generated
    
    ## Example
```json
    {
        "question": "What products are available?"
    }
```
    """
    
    try:
        # Use user_id from request if provided, otherwise use authenticated user's ID
        user_id = request.user_id if hasattr(request, 'user_id') and request.user_id else str(current_user.id)
        
        # Generate user-specific thread_id
        thread_id = f"user_{user_id}"
        
        logger.info(f"Chat request from user {current_user.username} (ID: {current_user.id})")
        logger.info(f"Question: {request.question[:100]}...")
        
        # Call the chat system with user_id
        response = await chat(
            question=request.question,
            thread_id=thread_id,
            user_id=user_id
        )
        
        logger.info(f"Response generated for user {current_user.username}")
        
        return ChatResponse(
            success=True,
            response=response,
            thread_id=thread_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                "question_length": len(request.question),
                "response_length": len(response),
                "user_id": current_user.id,
                "username": current_user.username
            }
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint for user {current_user.username}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Streaming chat endpoint (Protected - placeholder for future implementation)
    """
    raise HTTPException(
        status_code=501,
        detail="Streaming endpoint not yet implemented"
    )


@router.get("/chat/history/{thread_id}")
async def get_chat_history(
    thread_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get conversation history for a specific thread (Protected)
    
    Users can only access their own conversation history.
    """
    try:
        # Verify user owns this thread
        expected_thread_id = f"user_{current_user.id}"
        if thread_id != expected_thread_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this conversation"
            )
        
        # Get conversation history
        history = conversation_store.get_messages(thread_id)
        
        return {
            "success": True,
            "thread_id": thread_id,
            "message_count": len(history),
            "messages": history
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.delete("/chat/history/{thread_id}")
async def clear_chat_history(
    thread_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Clear conversation history for a specific thread (Protected)
    
    Users can only clear their own conversation history.
    """
    try:
        # Verify user owns this thread
        expected_thread_id = f"user_{current_user.id}"
        if thread_id != expected_thread_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to clear this conversation"
            )
        
        # Clear conversation history
        conversation_store.clear_thread(thread_id)
        
        logger.info(f"User {current_user.username} cleared chat history for thread {thread_id}")
        
        return {
            "success": True,
            "message": "Conversation history cleared successfully",
            "thread_id": thread_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/chat/my-history")
async def get_my_chat_history(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user's conversation history (Protected)
    
    Convenience endpoint to get authenticated user's chat history.
    """
    thread_id = f"user_{current_user.id}"
    history = conversation_store.get_messages(thread_id)
    
    return {
        "success": True,
        "thread_id": thread_id,
        "user_id": current_user.id,
        "username": current_user.username,
        "message_count": len(history),
        "messages": history
    }



# ==================== MAIN ====================
if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:router",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )
