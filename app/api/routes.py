from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
from app.models.email_store import EmailStore

router = APIRouter()

class EmailRequest(BaseModel):
    subject: str
    body: str
    mode: Optional[str] = "topic"  # "topic" or "similar"

class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str

class EmailStoreRequest(BaseModel):
    subject: str
    body: str
    ground_truth_topic: Optional[str] = None

class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]

class TopicCreateRequest(BaseModel):
    topic: str
    description: str

class TopicCreateResponse(BaseModel):
    message: str
    topic: str
    available_topics: List[str]

class EmailAddResponse(BaseModel):
    message: str
    email_id: str

@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
        result = inference_service.classify_email(email, mode=request.mode)
        
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}

@router.post("/topics", response_model=TopicCreateResponse)
async def create_topic(request: TopicCreateRequest):
    try:
        inference_service = EmailTopicInferenceService()
        created_topic = inference_service.add_new_topic(request.topic, request.description)
        info = inference_service.get_pipeline_info()

        return TopicCreateResponse(
            message="New topic added",
            topic=created_topic,
            available_topics=info["available_topics"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emails", response_model=EmailAddResponse)
async def store_email(request: EmailStoreRequest):
    try:
        email = Email(subject=request.subject, body=request.body)
        store = EmailStore()
        email_id = store.add_email(email, ground_truth_topic=request.ground_truth_topic)
        return EmailAddResponse(message="Email stored", email_id=email_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/info") 
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()
