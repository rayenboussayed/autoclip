from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .progress import progress_manager
from .routers import api, view
from .state import registry

# How often the lifespan task removes expired jobs. A module constant so tests can shrink it.
EVICT_SECONDS = 60.0

# OpenTelemetry setup
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    settings = get_settings()
    provider = TracerProvider(resource=Resource.create({"service.name": "autoclip-server"}))
    if settings.otp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otp_endpoint)))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    loop = asyncio.get_event_loop()
    progress_manager.bind_loop(loop)

    async def evictor() -> None:
        while True:
            await asyncio.sleep(EVICT_SECONDS)
            # Disk deletion must never block the event loop (or a websocket handshake behind it).
            await asyncio.to_thread(registry.evict_expired, settings.job_ttl_seconds)

    task = asyncio.ensure_future(evictor())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def create_app() -> FastAPI:
    app = FastAPI(title="autoclip server", version="2.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api.router)
    app.include_router(view.router)

    if _OTEL_AVAILABLE:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except Exception:  # noqa: S110
            pass
    return app


app = create_app()


def main() -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run("autoclip.server.main:app", host=settings.host, port=settings.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
