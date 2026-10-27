import logging
import os
import secrets
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

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from ticketlens.config import get_project_root, load_config
from ticketlens.predict import TicketPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.api")

# Global predictor instance loaded once at startup
predictor: Optional[TicketPredictor] = None

API_KEY_ENV = "TICKETLENS_API_KEY"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description=f"Required only when {API_KEY_ENV} is set on the server.")


def _configured_api_key() -> str:
    return os.environ.get(API_KEY_ENV, "").strip()


def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    """Enforce the X-API-Key header when TICKETLENS_API_KEY is configured; open demo mode otherwise."""
    expected = _configured_api_key()
    if not expected:
        return
    if not api_key or not secrets.compare_digest(api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Provide it in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


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
    # The API uses an X-API-Key header, not cookies, so credentialed CORS is unnecessary.
    allow_credentials=False,
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
    itil_cues: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="ITIL keyword cues that adjusted the model probabilities, grouped by category.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Input quality warnings (e.g. non-English text). Any warning forces 'Needs Review'.",
    )
    routing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Where the ticket goes: resolver queue and response target (auto-routed), or the human "
        "triage queue with the model's suggested queue (Needs Review). Configured under `routing` in config.yaml.",
    )
    processed_text: str = Field(
        "",
        description="The exact text the model classified, after cleaning and (if enabled) PII redaction.",
    )


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
    auth_required: bool


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["General"])
@app.api_route("/demo", methods=["GET", "HEAD"], response_class=HTMLResponse, tags=["General"])
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


@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    tags=["Monitoring"],
    responses={503: {"model": HealthResponse, "description": "No model loaded (deploy should not go live)"}},
)
async def health_check(response: Response):
    """Health check: 200 when a model is ready, 503 otherwise so Render keeps the previous deploy live."""
    global predictor
    is_ready = predictor is not None and predictor.is_ready
    loaded = predictor.get_loaded_models() if predictor else []
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
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
    default_model = cfg.get("inference", {}).get("default_model", "transformer")
    if loaded and default_model not in loaded:
        # Report the model that will actually serve requests after fallback.
        default_model = loaded[0]
    return MetadataResponse(
        project_name=cfg.get("project_name", "TicketLens"),
        version=cfg.get("version", "0.1.0"),
        dataset_name=cfg.get("dataset", {}).get("name", "Tobi-Bueck/customer-support-tickets"),
        target_categories=["Incident", "Request", "Problem", "Change"],
        device=str(predictor.device) if predictor else "cpu",
        models_available=loaded,
        default_model=default_model,
        confidence_threshold=float(cfg.get("inference", {}).get("confidence_threshold", 0.80)),
        auth_required=bool(_configured_api_key()),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"], dependencies=[Depends(require_api_key)])
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
