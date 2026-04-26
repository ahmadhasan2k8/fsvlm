"""FastAPI REST API for FSVLM inference.

Endpoints:
    POST /inspect  — inspect an uploaded image
    GET  /health   — health check
    GET  /adapters — list available adapters
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fsvlm.config import FSVLMConfig

# These must be importable without GPU — they only use stdlib + starlette
_fastapi_imported = False


def _ensure_fastapi() -> None:
    global _fastapi_imported
    if _fastapi_imported:
        return
    try:
        import fastapi  # noqa: F401

        _fastapi_imported = True
    except ImportError:
        raise ImportError("FastAPI not installed. Install with: pip install 'fsvlm[serve]'")


def create_app(
    config: FSVLMConfig | None = None,
    adapter_path: Path | None = None,
) -> Any:
    """Create and configure the FastAPI application.

    Args:
        config: FSVLM configuration.
        adapter_path: Path to adapter to load on startup.

    Returns:
        Configured FastAPI app instance.
    """
    _ensure_fastapi()

    # Import at call time, not module level — keeps `import fsvlm` fast
    import fastapi
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    from fsvlm.agents.inspector_agent import InspectorAgent

    if config is None:
        from fsvlm.config import load_config

        config = load_config()

    # Shared inspector instance — loaded once, reused across requests
    inspector = InspectorAgent(config)
    _state: dict[str, Path | None] = {"adapter_path": adapter_path}

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):  # type: ignore[no-untyped-def]
        if _state["adapter_path"] is not None:
            inspector.load_adapter(_state["adapter_path"])
        yield
        inspector.unload()

    app = fastapi.FastAPI(
        title="FSVLM",
        description="Visual defect detection API powered by Gemma 4 VLM",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {
            "status": "ok",
            "model_loaded": inspector.is_loaded,
            "adapter": str(_state["adapter_path"]) if _state["adapter_path"] else None,
        }

    @app.post("/inspect")
    async def inspect_image(request: fastapi.Request) -> JSONResponse:
        """Inspect an uploaded image for defects.

        Send image as multipart form data with field name 'file'.
        Optional 'threshold' field (default 0.5).
        """
        if not inspector.is_loaded:
            return JSONResponse(
                {"detail": "No adapter loaded. Start server with --adapter flag."},
                status_code=503,
            )

        form = await request.form()
        file = form.get("file")
        if file is None:
            return JSONResponse(
                {"detail": "No file uploaded. Send image as 'file' field."},
                status_code=400,
            )

        threshold = float(form.get("threshold", 0.5))

        # Save upload to temp file
        filename = getattr(file, "filename", "image.png") or "image.png"
        suffix = Path(filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = inspector.inspect(tmp_path, threshold=threshold)
            return JSONResponse(
                {
                    "pass_fail": "PASS" if result.pass_fail else "FAIL",
                    "confidence": round(result.confidence, 4),
                    "description": result.description,
                    "inference_time_ms": round(result.inference_time_ms, 1),
                    "model_name": result.model_name,
                    "adapter_name": result.adapter_name,
                }
            )
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/adapters")
    async def list_adapters() -> list[dict]:
        """List available adapters."""
        from fsvlm.models.adapter import METADATA_FILENAME, load_adapter_metadata

        adapters_dir = config.adapters_dir
        results: list[dict] = []

        if not adapters_dir.exists():
            return results

        # Search for adapters (up to 2 levels deep)
        for d in adapters_dir.iterdir():
            if not d.is_dir():
                continue
            meta_path = d / METADATA_FILENAME
            if meta_path.exists():
                try:
                    meta = load_adapter_metadata(d)
                    results.append(
                        {
                            "path": str(d),
                            "name": meta.adapter_name,
                            "base_model": meta.base_model,
                            "version": meta.adapter_version,
                            "created_at": meta.created_at,
                            "f1": meta.validation_metrics.f1 if meta.validation_metrics else None,
                        }
                    )
                except Exception:
                    pass
            # Check one level deeper
            for child in d.iterdir():
                if child.is_dir() and (child / METADATA_FILENAME).exists():
                    try:
                        meta = load_adapter_metadata(child)
                        results.append(
                            {
                                "path": str(child),
                                "name": meta.adapter_name,
                                "base_model": meta.base_model,
                                "version": meta.adapter_version,
                                "created_at": meta.created_at,
                                "f1": meta.validation_metrics.f1 if meta.validation_metrics else None,
                            }
                        )
                    except Exception:
                        pass

        return results

    @app.post("/load-adapter")
    async def load_adapter(adapter_dir: str) -> dict:
        """Load a different adapter at runtime."""
        path = Path(adapter_dir)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Adapter not found: {path}")

        inspector.unload()
        inspector.load_adapter(path)
        _state["adapter_path"] = path

        return {"status": "loaded", "adapter": str(path)}

    return app


def run_server(
    config: FSVLMConfig | None = None,
    adapter_path: Path | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the FastAPI server with uvicorn.

    Args:
        config: FSVLM configuration.
        adapter_path: Path to adapter to load on startup.
        host: Server bind host.
        port: Server bind port.
    """
    import uvicorn

    if config is None:
        from fsvlm.config import load_config

        config = load_config()

    app = create_app(config=config, adapter_path=adapter_path)
    uvicorn.run(
        app,
        host=host or config.serve_host,
        port=port or config.serve_port,
        log_level="info",
    )
