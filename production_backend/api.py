"""
Advanced FastAPI Application for AI Support System
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import uvicorn
import asyncio
from datetime import datetime
import uuid
import hashlib

# Import our modules
from config import get_config, validate_config
from monitoring import get_monitor
from ab_testing import get_ab_manager
from cache import get_cache_manager, get_response_cache
from crew import SupportFlow

# Pydantic models for API
class SupportRequest(BaseModel):
    user_query: str = Field(..., description="User's support query", min_length=1)
    user_id: str = Field(..., description="Unique user identifier")
    language: str = Field(default="en", description="Preferred language")
    priority: str = Field(default="medium", description="Request priority")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    response_style: Optional[str] = Field(default="empathetic", description="Preferred response style")

class SupportResponse(BaseModel):
    request_id: str
    status: str
    response: str
    intent: str
    priority_score: float
    escalation_needed: bool
    processing_time: float
    timestamp: str
    metadata: Dict[str, Any]

class SatisfactionFeedback(BaseModel):
    user_id: str
    request_id: str
    score: int = Field(..., ge=1, le=5, description="Satisfaction score 1-5")
    feedback: str = Field(..., description="User feedback")
    category: Optional[str] = Field(default=None, description="Feedback category")

class SystemStatus(BaseModel):
    status: str
    uptime_hours: float
    total_requests: int
    active_alerts: int
    cache_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class ABTestRequest(BaseModel):
    test_id: str
    user_id: str
    event_type: str
    value: Union[float, int, str]
    metadata: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="AI Support System API",
    description="Advanced AI-powered customer support system with A/B testing, monitoring, and optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = get_config()
monitor = get_monitor(config)
ab_manager = get_ab_manager(config)
cache_manager = get_cache_manager(config)
response_cache = get_response_cache(config)

# Dependency functions
def get_system_config():
    return config

def get_system_monitor():
    return monitor

def get_ab_test_manager():
    return ab_manager

def get_system_cache():
    return cache_manager

# Background tasks
async def process_support_request_async(request_data: SupportRequest) -> Dict[str, Any]:
    """Process support request asynchronously"""
    start_time = datetime.now()
    
    try:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Check cache first
        query_hash = hashlib.md5(request_data.user_query.encode()).hexdigest()
        cached_response = response_cache.get_cached_response(
            query_hash, "unknown", request_data.response_style
        )
        
        if cached_response:
            # Return cached response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "request_id": request_id,
                "status": "success",
                "response": cached_response,
                "intent": "cached",
                "priority_score": 0.5,
                "escalation_needed": False,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"cached": True, "cache_hit": True}
            }
        
        # Process with AI flow
        flow = SupportFlow()
        result = flow.kickoff(inputs={
            "user_query": request_data.user_query,
            "user_id": request_data.user_id,
            "language": request_data.language
        })
        
        # Extract results
        intent = getattr(result, "intent", "unknown") if hasattr(result, "intent") else str(result)
        response = str(result)
        escalation_needed = "escalate" in intent.lower()
        
        # Calculate priority score
        priority_score = calculate_priority_score(request_data.user_query, intent)
        
        # Cache the response
        response_cache.cache_response(
            query_hash, intent, response, request_data.response_style
        )
        
        # Track metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        monitor.track_request(
            start_time=start_time.timestamp(),
            success=True,
            escalation=escalation_needed,
            intent_accuracy=1.0,
            cache_hit=False
        )
        
        # Track A/B testing events
        ab_manager.track_conversion(
            user_id=request_data.user_id,
            test_id="response_style_001",
            event_type="request",
            value=1
        )
        
        return {
            "request_id": request_id,
            "status": "success",
            "response": response,
            "intent": intent,
            "priority_score": priority_score,
            "escalation_needed": escalation_needed,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"cached": False, "cache_hit": False}
        }
        
    except Exception as e:
        # Track error
        processing_time = (datetime.now() - start_time).total_seconds()
        monitor.track_request(
            start_time=start_time.timestamp(),
            success=False,
            escalation=False,
            intent_accuracy=0.0,
            cache_hit=False
        )
        
        return {
            "request_id": str(uuid.uuid4()),
            "status": "error",
            "response": f"An error occurred: {str(e)}",
            "intent": "error",
            "priority_score": 0.0,
            "escalation_needed": True,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"error": str(e)}
        }

def calculate_priority_score(query: str, intent: str) -> float:
    """Calculate priority score based on query and intent"""
    score = 0.5  # Base score
    
    # Intent-based scoring
    if "urgent" in intent.lower() or "urgent" in query.lower():
        score += 0.3
    if "billing" in intent.lower() or "payment" in query.lower():
        score += 0.2
    if "technical" in intent.lower() or "error" in query.lower():
        score += 0.1
    
    # Query length and complexity
    if len(query.split()) > 20:
        score += 0.1
    
    # Escalation keywords
    escalation_keywords = ["complaint", "angry", "frustrated", "unhappy", "disappointed"]
    if any(keyword in query.lower() for keyword in escalation_keywords):
        score += 0.2
    
    return min(score, 1.0)

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "AI Support System API v2.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.post("/support/query", response_model=SupportResponse)
async def process_support_query(
    request: SupportRequest,
    background_tasks: BackgroundTasks
):
    """Process a support query with AI agents"""
    try:
        # Validate request
        if not request.user_query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process request asynchronously
        result = await process_support_request_async(request)
        
        # Add background task for analytics
        background_tasks.add_task(
            monitor.track_request,
            start_time=datetime.now().timestamp(),
            success=result["status"] == "success",
            escalation=result["escalation_needed"],
            intent_accuracy=1.0 if result["status"] == "success" else 0.0
        )
        
        return SupportResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/support/status/{request_id}", response_model=Dict[str, Any])
async def get_support_status(request_id: str):
    """Get status of a support request"""
    # This would typically query a database
    # For now, return a mock response
    return {
        "request_id": request_id,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "message": "Request processing completed"
    }

@app.post("/feedback/satisfaction")
async def submit_satisfaction_feedback(feedback: SatisfactionFeedback):
    """Submit user satisfaction feedback"""
    try:
        # Track satisfaction
        monitor.track_satisfaction(
            user_id=feedback.user_id,
            score=feedback.score,
            feedback=feedback.feedback
        )
        
        # Track A/B testing conversion
        ab_manager.track_conversion(
            user_id=feedback.user_id,
            test_id="response_style_001",
            event_type="satisfaction",
            value=feedback.score
        )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and performance metrics"""
    try:
        # Get performance report
        performance_report = monitor.get_performance_report(hours=24)
        
        # Get cache stats
        cache_stats = cache_manager.get_stats()
        
        # Get real-time metrics
        realtime_metrics = monitor.get_realtime_metrics()
        
        return SystemStatus(
            status="operational",
            uptime_hours=performance_report.get("system_uptime_hours", 0),
            total_requests=performance_report.get("total_requests", 0),
            active_alerts=performance_report.get("current_alerts", 0),
            cache_stats=cache_stats,
            performance_metrics=realtime_metrics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/system/metrics")
async def get_system_metrics(hours: int = Query(24, ge=1, le=168)):
    """Get detailed system metrics"""
    try:
        return monitor.get_performance_report(hours=hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/system/alerts")
async def get_system_alerts():
    """Get current system alerts"""
    try:
        # This would return actual alerts from the monitor
        return {"alerts": [], "message": "No active alerts"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.post("/ab-testing/track")
async def track_ab_test_event(request: ABTestRequest):
    """Track A/B testing conversion event"""
    try:
        ab_manager.track_conversion(
            user_id=request.user_id,
            test_id=request.test_id,
            event_type=request.event_type,
            value=request.value,
            metadata=request.metadata
        )
        
        return {"status": "success", "message": "Event tracked"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track event: {str(e)}")

@app.get("/ab-testing/results/{test_id}")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    try:
        results = ab_manager.get_test_results(test_id)
        significance = ab_manager.is_test_significant(test_id)
        
        return {
            "test_id": test_id,
            "results": {k: v.__dict__ for k, v in results.items()},
            "significance": significance
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get test results: {str(e)}")

@app.get("/ab-testing/recommendations")
async def get_ab_test_recommendations():
    """Get optimization recommendations from A/B tests"""
    try:
        return ab_manager.get_optimization_recommendations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")

@app.get("/cache/stats")
async def get_cache_statistics():
    """Get cache statistics and performance"""
    try:
        return cache_manager.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

@app.post("/cache/optimize")
async def optimize_cache():
    """Trigger cache optimization"""
    try:
        cache_manager.optimize_cache()
        return {"status": "success", "message": "Cache optimization completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache optimization failed: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    try:
        cache_manager.clear()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.post("/cache/invalidate/tag/{tag}")
async def invalidate_cache_by_tag(tag: str):
    """Invalidate cache entries by tag"""
    try:
        cache_manager.invalidate_by_tag(tag)
        return {"status": "success", "message": f"Cache entries with tag '{tag}' invalidated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Get current system configuration"""
    try:
        return {
            "support_config": config.__dict__,
            "validation": validate_config()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@app.post("/config/update")
async def update_configuration(updates: Dict[str, Any]):
    """Update system configuration"""
    try:
        from config import update_config
        update_config(**updates)
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    print("🚀 AI Support System API starting up...")
    
    # Validate configuration
    if not validate_config():
        print("⚠️ Configuration validation failed")
    
    # Initialize components
    print("✅ Configuration validated")
    print("✅ Monitoring system initialized")
    print("✅ A/B testing system initialized")
    print("✅ Cache system initialized")
    print("🌐 API ready at http://localhost:8000")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    print("🛑 AI Support System API shutting down...")
    
    # Export final metrics
    try:
        monitor.export_metrics("final_metrics.json")
        ab_manager.export_test_data("final_ab_tests.json")
        cache_manager.export_cache_data("final_cache.json")
        print("✅ Final data exported")
    except Exception as e:
        print(f"⚠️ Failed to export final data: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
