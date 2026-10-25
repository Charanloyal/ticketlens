import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Ensure package directory and root are in sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from ticketlens.config import get_device, get_project_root, load_config
from ticketlens.predict import TicketPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.api")

# Global predictor instance loaded once at startup
predictor: Optional[TicketPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load models once during startup."""
    global predictor
    logger.info("Initializing TicketLens Predictor at startup...")
    cfg = load_config()
    predictor = TicketPredictor(cfg=cfg)
    logger.info("Startup complete. Loaded models: %s", predictor.get_loaded_models())
    yield
    logger.info("Shutting down TicketLens API service.")


app = FastAPI(
    title="TicketLens IT Ticket Classification API",
    description="Production API for classifying IT support tickets using fine-tuned DistilBERT (LoRA) and TF-IDF baseline models.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount reports directory for figures and visualizations if present
reports_dir = get_project_root() / "reports"
if reports_dir.exists():
    app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The IT ticket content, subject, or description to classify.",
        examples=["Cannot connect to corporate VPN. TLS handshake error 504."],
    )
    model_type: Optional[Literal["transformer", "baseline"]] = Field(
        None,
        description="Optional model selector. Defaults to the configured default model (transformer).",
    )
    redact_pii: bool = Field(
        True,
        description="Whether to redact detected email addresses, phone numbers, and IPs before classification.",
    )

    @field_validator("text")
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ticket text must not be empty or whitespace only.")
        return v.strip()


class PredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    decision: str
    probabilities: Dict[str, float]
    model_type: str
    model_version: str
    confidence_threshold: float


class HealthResponse(BaseModel):
    status: str
    ready: bool
    models_loaded: List[str]


class MetadataResponse(BaseModel):
    project_name: str
    version: str
    dataset_name: str
    target_categories: List[str]
    device: str
    models_available: List[str]
    default_model: str
    confidence_threshold: float


@app.get("/", response_class=HTMLResponse, tags=["General"])
@app.get("/demo", response_class=HTMLResponse, tags=["General"])
async def root(request: Request, format: Optional[str] = None):
    """Serve the interactive TicketLens live demo dashboard, or JSON status if requested."""
    accept_header = request.headers.get("accept", "")
    if format == "json" or ("application/json" in accept_header and "text/html" not in accept_header):
        return JSONResponse({
            "service": "TicketLens IT Ticket Classification API",
            "version": "0.1.0",
            "status": "healthy",
            "documentation": "/docs",
            "health": "/health",
            "metadata": "/metadata",
        })

    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return HTMLResponse("<h1>TicketLens API Operational</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint confirming service status and loaded model availability."""
    global predictor
    is_ready = predictor is not None and predictor.is_ready
    loaded = predictor.get_loaded_models() if predictor else []
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        ready=is_ready,
        models_loaded=loaded,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["Metadata"])
async def get_metadata():
    """Retrieve system configuration, model version, and target category information."""
    cfg = load_config()
    loaded = predictor.get_loaded_models() if predictor else []
    return MetadataResponse(
        project_name=cfg.get("project_name", "TicketLens"),
        version=cfg.get("version", "0.1.0"),
        dataset_name=cfg.get("dataset", {}).get("name", "Tobi-Bueck/customer-support-tickets"),
        target_categories=["Incident", "Request", "Problem", "Change"],
        device=str(get_device()),
        models_available=loaded,
        default_model=cfg.get("inference", {}).get("default_model", "transformer"),
        confidence_threshold=float(cfg.get("inference", {}).get("confidence_threshold", 0.80)),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_ticket(request: PredictRequest):
    """
    Classify an incoming IT support ticket.
    Returns predicted category, probabilities, confidence score, and automated routing decision.
    """
    global predictor
    if predictor is None or not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model artifacts found on server. "
                "Please run 'python -m ticketlens.train_baseline' or 'python -m ticketlens.train_transformer' to generate models."
            ),
        )

    try:
        result = predictor.predict(
            text=request.text,
            model_type=request.model_type,
            redact_pii=request.redact_pii,
        )
        return PredictResponse(**result)
    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}",
        )
