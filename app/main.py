"""Stateless API serving the frozen ChurnGuard calibrated model."""

import logging
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import PUBLIC_NAME, PUBLIC_TAGLINE, Settings
from app.inference import customers_to_frame, predict_customers
from app.landing import render_landing_page
from app.metrics import INFERENCE_PATHS, ServiceMetrics
from app.model_service import ModelService
from app.monitoring import check_drift, load_reference
from app.ranking import rank_predictions
from app.schemas import (
    BatchPrediction,
    BatchRequest,
    Customer,
    Prediction,
    RankingResult,
    RankRequest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    metrics = ServiceMetrics()

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        logger.info("Starting %s API", PUBLIC_NAME)
        api.state.service = None
        api.state.reference = None
        try:
            service = ModelService.load(settings)
            reference = load_reference(settings.reference_path)
            api.state.service = service
            api.state.reference = reference
        except Exception:
            # Deliberately omit exception text: deserializer errors can contain private paths.
            logger.error("Startup verification failed; service remains unready")
        yield
        api.state.service = None
        api.state.reference = None

    api = FastAPI(
        title=f"{PUBLIC_NAME} — {PUBLIC_TAGLINE}",
        description=(
            "Calibrated churn scoring, capacity-constrained retention ranking, and "
            "drift monitoring over a frozen, checksum-verified model artifact."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    api.state.service = None
    api.state.reference = None
    api.state.metrics = metrics

    @api.middleware("http")
    async def observe_inference(request: Request, call_next):
        path = request.url.path
        if path not in INFERENCE_PATHS or request.method != "POST":
            return await call_next(request)
        start = perf_counter()
        metrics.requests.labels(endpoint=path).inc()
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                metrics.errors.labels(endpoint=path).inc()
            return response
        except Exception:
            metrics.errors.labels(endpoint=path).inc()
            logger.error("Inference request failed on %s", path)
            return JSONResponse(status_code=500, content={"detail": "Internal service error"})
        finally:
            metrics.latency.labels(endpoint=path).observe(perf_counter() - start)

    @api.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Do not echo input values, including nonfinite JSON numbers or entire batches.
        errors = [
            {"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": errors})

    def require_service() -> ModelService:
        if api.state.service is None:
            raise HTTPException(status_code=503, detail="Model service is not ready")
        return api.state.service

    def score(customers: list[Customer]) -> list[Prediction]:
        service = require_service()
        metrics.batch_size.observe(len(customers))
        try:
            predictions = predict_customers(service, customers)
        except Exception:
            logger.error("Model inference failed")
            raise HTTPException(status_code=503, detail="Model inference unavailable") from None
        metrics.predictions.inc(len(predictions))
        return predictions

    @api.get("/", response_class=HTMLResponse, include_in_schema=False)
    def landing():
        """Recruiter-facing overview; renders generated, artifact-derived facts."""
        return HTMLResponse(render_landing_page(settings.landing_facts_path))

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.get("/ready")
    def ready():
        service = require_service()
        return {
            "status": "ready",
            "model_version": service.metadata["model_version"],
            "model_loaded": True,
            "metadata_loaded": True,
            "checksum_verified": service.checksum_verified,
        }

    @api.get("/model-info")
    def model_info():
        return require_service().public_info()

    @api.post("/predict", response_model=Prediction)
    def predict(customer: Customer):
        """Return predicted churn probability for one customer."""
        return score([customer])[0]

    @api.post("/batch-predict", response_model=BatchPrediction)
    def batch_predict(batch: BatchRequest):
        return BatchPrediction(
            predictions=score(batch.customers), total_customers=len(batch.customers)
        )

    @api.post("/rank", response_model=RankingResult)
    def rank(request: RankRequest):
        """Predictive prioritization; does not estimate the causal benefit of outreach."""
        return rank_predictions(score(request.customers), request)

    @api.post("/monitor/drift")
    def drift(batch: BatchRequest):
        require_service()
        try:
            result = check_drift(customers_to_frame(batch.customers), api.state.reference)
        except Exception:
            logger.error("Drift calculation failed")
            raise HTTPException(status_code=503, detail="Drift monitoring unavailable") from None
        metrics.drift_checks.inc()
        if result["warnings"]:
            logger.warning(
                "Drift warning: %s (%d rows)", result["overall_status"], len(batch.customers)
            )
        return result

    @api.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(
            content=generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    return api


app = create_app()
