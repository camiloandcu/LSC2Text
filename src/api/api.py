from __future__ import annotations

import argparse
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from scripts.infer import DEFAULT_MODEL_PATH, DEFAULT_TOP_K, run_inference


@dataclass(slots=True)
class ApiSettings:
    model_path: Path = Path(DEFAULT_MODEL_PATH)
    top_k: int = DEFAULT_TOP_K
    service_name: str = "LSC2Text Inference API"
    service_version: str = "0.1.0"


class PredictionItem(BaseModel):
    label: str
    confidence: float = Field(ge=0.0)


class PredictResponse(BaseModel):
    timestamp: str
    model: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str = "ok"
    ready: bool = True


class MetadataResponse(BaseModel):
    service: str
    version: str
    default_model_path: str
    top_k: int
    endpoints: list[str]


TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _predict_from_image_path(image_path: Path, settings: ApiSettings) -> PredictResponse:
    return PredictResponse.model_validate(
        run_inference(
            image_path,
            settings.model_path,
            top_k=settings.top_k,
        )
    )


def _format_predictions(predictions: list[PredictionItem]) -> list[dict[str, str]]:
    return [
        {
            "label": prediction.label,
            "confidence_percent": f"{prediction.confidence * 100:.2f}%",
        }
        for prediction in predictions
    ]


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    app = FastAPI(title=settings.service_name, version=settings.service_version)
    app.state.settings = settings

    @app.exception_handler(HTTPException)
    def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "request_error", "details": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    def validation_exception_handler(_, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "details": exc.errors()},
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {
                "service_name": settings.service_name,
                "default_model_path": str(settings.model_path),
                "top_k": settings.top_k,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        return MetadataResponse(
            service=settings.service_name,
            version=settings.service_version,
            default_model_path=str(settings.model_path),
            top_k=settings.top_k,
            endpoints=["GET /health", "GET /metadata", "POST /predict"],
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(image: UploadFile = File(...)) -> PredictResponse:
        if not image.filename:
            raise HTTPException(status_code=400, detail="uploaded file must include a filename")

        raw_bytes = image.file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="uploaded file is empty")

        suffix = Path(image.filename).suffix or ".img"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(raw_bytes)
                temp_path = Path(handle.name)

            return _predict_from_image_path(temp_path, settings)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image file: {exc}") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @app.post("/frontend/predict", response_class=HTMLResponse)
    def predict_frontend(request: Request, image: UploadFile = File(...), submit_label: str = Form("Upload image")) -> HTMLResponse:
        del submit_label
        if not image.filename:
            return templates.TemplateResponse(
                request,
                "error.html",
                {
                    "service_name": settings.service_name,
                    "message": "Please choose a valid image file before submitting.",
                },
                status_code=400,
            )

        raw_bytes = image.file.read()
        if not raw_bytes:
            return templates.TemplateResponse(
                request,
                "error.html",
                {
                    "service_name": settings.service_name,
                    "message": "The uploaded file was empty or unreadable.",
                },
                status_code=400,
            )

        suffix = Path(image.filename).suffix or ".img"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(raw_bytes)
                temp_path = Path(handle.name)

            prediction = _predict_from_image_path(temp_path, settings)
            image_data_b64 = base64.b64encode(raw_bytes).decode("utf-8")
            return templates.TemplateResponse(
                request,
                "result.html",
                {
                    "service_name": settings.service_name,
                    "image_data": image_data_b64,
                    "default_model_path": str(settings.model_path),
                    "predictions": _format_predictions(prediction.predictions),
                },
            )
        except FileNotFoundError:
            return templates.TemplateResponse(
                request,
                "error.html",
                {
                    "service_name": settings.service_name,
                    "message": "The configured model could not be found.",
                },
                status_code=503,
            )
        except Exception:
            return templates.TemplateResponse(
                request,
                "error.html",
                {
                    "service_name": settings.service_name,
                    "message": "The uploaded file is not a valid image.",
                },
                status_code=400,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local FastAPI inference backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help=f"Default model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Default top-k predictions (default: {DEFAULT_TOP_K})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ApiSettings(model_path=Path(args.model_path), top_k=args.top_k)
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


app = create_app()


if __name__ == "__main__":
    main()