"""
Pydantic Schemas for FastAPI Request & Response Models.
"""

from pydantic import BaseModel, Field, model_validator


class EmailPredictRequest(BaseModel):
    """
    Request payload schema for email phishing prediction.
    """
    subject: str = Field(default="", description="Email subject line")
    body: str = Field(default="", description="Email body text")


class ProbabilitiesSchema(BaseModel):
    """
    Schema for class probabilities output.
    """
    LEGITIMATE: float = Field(..., ge=0.0, le=1.0, description="Probability of legitimate email")
    PHISHING: float = Field(..., ge=0.0, le=1.0, description="Probability of phishing email")


class EmailPredictResponse(BaseModel):
    """
    Response payload schema for email phishing prediction.
    """
    prediction: str = Field(..., description="Predicted label: PHISHING or LEGITIMATE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    probabilities: ProbabilitiesSchema


class HealthCheckResponse(BaseModel):
    """
    Response payload schema for API health check endpoint.
    """
    status: str = Field(default="healthy", description="API health status")
    model_loaded: bool = Field(..., description="Indicates if BERT predictor model is loaded")
