"""
FastAPI REST API Server for Phishing Email Detection BERT Model.
Exposes GET /api/v1/health and POST /api/v1/predict endpoints.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root directory is in sys.path for direct execution and IDE linting
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictor import PhishingPredictor
from src.backend.schemas import (
    EmailPredictRequest,
    EmailPredictResponse,
    ProbabilitiesSchema,
    HealthCheckResponse
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load PhishingPredictor once at startup.
    """
    print("[*] FastAPI Lifespan: Initializing PhishingPredictor...")
    try:
        app.state.predictor = PhishingPredictor()
        print("[+] FastAPI Lifespan: PhishingPredictor successfully loaded into app.state!")
    except Exception as e:
        print(f"[-] FastAPI Lifespan Warning: Failed to load predictor model: {e}")
        app.state.predictor = None
    yield
    print("[*] FastAPI Lifespan: Shutting down API server...")


app = FastAPI(
    title="Phishing Email Detection API",
    description="REST API service powered by fine-tuned BERT for detecting phishing emails.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    """
    Custom exception handler for validation errors (e.g., empty subject & body).
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


def get_predictor(request: Request) -> PhishingPredictor:
    """
    Dependency helper to retrieve loaded PhishingPredictor instance from app state.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor model is not loaded or currently unavailable."
        )
    return predictor


@app.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    summary="API Health & Model Status",
    tags=["Health"]
)
async def health_check(request: Request):
    """
    Returns API health status and whether the BERT predictor model is loaded.
    """
    predictor = getattr(request.app.state, "predictor", None)
    model_loaded = predictor is not None
    return HealthCheckResponse(
        status="healthy",
        model_loaded=model_loaded
    )


@app.post(
    "/api/v1/predict",
    response_model=EmailPredictResponse,
    summary="Predict Phishing Risk for an Email",
    tags=["Prediction"]
)
async def predict_email(
    payload: EmailPredictRequest,
    predictor: PhishingPredictor = Depends(get_predictor)
):
    """
    Analyzes an email subject and body, returning classification label and confidence score.
    """
    # Validation check: Ensure at least one of subject or body has non-whitespace characters
    subject_clean = payload.subject.strip() if payload.subject else ""
    body_clean = payload.body.strip() if payload.body else ""

    if not subject_clean and not body_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least subject or body must be non-empty."
        )

    try:
        result = predictor.predict(subject=payload.subject, body=payload.body)
        return EmailPredictResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=ProbabilitiesSchema(
                LEGITIMATE=result["probabilities"]["LEGITIMATE"],
                PHISHING=result["probabilities"]["PHISHING"]
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend.main:app", host="127.0.0.1", port=8000, reload=True)
