"""
server.py — FastAPI inference server with dynamic batching and Prometheus metrics.

Endpoints:
  POST /predict       — single inference request
  POST /batch_predict — explicit batch inference
  GET  /health        — server status
  GET  /metrics       — Prometheus-format metrics

Run with:
  uvicorn server:app --host 0.0.0.0 --port 8000

Or run directly (uses uvicorn programmatically):
  python3 src/server.py
"""

import time
import queue
import threading
import sys
import os
from typing import List, Optional

import numpy as np

# Allow local imports
sys.path.insert(0, os.path.dirname(__file__))

from model import MLP, get_or_create_model
from batching import BatchAccumulator

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ServerMetrics:
    """Thread-safe server metrics."""

    def __init__(self):
        self._lock              = threading.Lock()
        self.requests_total     = 0
        self.requests_error     = 0
        self.latencies_ms: list = []     # keep last 10000
        self.batch_sizes: list  = []     # keep last 10000
        self.start_time         = time.time()

    def record(self, latency_ms: float, batch_size: int = 1, error: bool = False):
        with self._lock:
            self.requests_total += 1
            if error:
                self.requests_error += 1
            self.latencies_ms.append(latency_ms)
            self.batch_sizes.append(batch_size)
            # Trim to keep memory bounded
            if len(self.latencies_ms) > 10000:
                self.latencies_ms = self.latencies_ms[-5000:]
                self.batch_sizes  = self.batch_sizes[-5000:]

    def percentile(self, p: float) -> float:
        with self._lock:
            if not self.latencies_ms:
                return 0.0
            return float(np.percentile(self.latencies_ms, p))

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def throughput_rps(self) -> float:
        up = self.uptime_seconds()
        return self.requests_total / max(up, 1e-9)

    def prometheus_text(self) -> str:
        p50  = self.percentile(50)
        p95  = self.percentile(95)
        p99  = self.percentile(99)
        mean_batch = float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0
        lines = [
            "# HELP inference_requests_total Total inference requests",
            "# TYPE inference_requests_total counter",
            f"inference_requests_total {self.requests_total}",
            "",
            "# HELP inference_errors_total Total errored requests",
            "# TYPE inference_errors_total counter",
            f"inference_errors_total {self.requests_error}",
            "",
            "# HELP inference_latency_ms_p50 Latency p50 (ms)",
            "# TYPE inference_latency_ms_p50 gauge",
            f"inference_latency_ms_p50 {p50:.3f}",
            "",
            "# HELP inference_latency_ms_p95 Latency p95 (ms)",
            "# TYPE inference_latency_ms_p95 gauge",
            f"inference_latency_ms_p95 {p95:.3f}",
            "",
            "# HELP inference_latency_ms_p99 Latency p99 (ms)",
            "# TYPE inference_latency_ms_p99 gauge",
            f"inference_latency_ms_p99 {p99:.3f}",
            "",
            "# HELP inference_mean_batch_size Mean batch size",
            "# TYPE inference_mean_batch_size gauge",
            f"inference_mean_batch_size {mean_batch:.2f}",
            "",
            "# HELP inference_throughput_rps Throughput (requests/second)",
            "# TYPE inference_throughput_rps gauge",
            f"inference_throughput_rps {self.throughput_rps():.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global singletons (initialized at startup)
# ---------------------------------------------------------------------------

_model:      Optional[MLP]              = None
_accumulator: Optional[BatchAccumulator] = None
_metrics:    ServerMetrics              = ServerMetrics()
_max_queue_size: int                    = 256
_model_path: str                        = "/tmp/inference_model"


def _get_model() -> MLP:
    global _model
    if _model is None:
        _model = get_or_create_model(_model_path + ".npz", [64, 128, 64, 10])
    return _model


def _get_accumulator() -> BatchAccumulator:
    global _accumulator
    if _accumulator is None:
        m = _get_model()
        _accumulator = BatchAccumulator(
            model_fn=m.predict,
            max_batch_size=32,
            max_wait_ms=10.0,
            max_queue_size=_max_queue_size,
        )
    return _accumulator


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    # Minimal stub so the module can still be imported for testing
    class BaseModel:  # type: ignore
        pass


if _HAS_FASTAPI:
    app = FastAPI(
        title="ML Inference Server",
        description="Dynamic-batching inference server for MLP model",
        version="1.0.0",
    )

    # --- Request / Response models ---

    class PredictRequest(BaseModel):
        inputs: List[List[float]]   # list of input vectors

    class PredictResponse(BaseModel):
        predictions: List[List[float]]
        latency_ms:  float
        batch_size:  int

    class HealthResponse(BaseModel):
        status:           str
        model:            str
        requests_served:  int
        uptime_seconds:   float
        queue_size:       int
        throughput_rps:   float

    # --- Startup ---

    @app.on_event("startup")
    async def startup():
        _get_accumulator()   # initialise background thread

    # --- Endpoints ---

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        """
        Single or small-batch inference with dynamic batching.
        Each request is submitted individually; the accumulator groups
        concurrent requests into larger batches automatically.
        """
        t_start = time.perf_counter()
        acc = _get_accumulator()

        inputs = np.array(req.inputs, dtype=np.float32)  # (N, in_features)

        try:
            if inputs.shape[0] == 1:
                result = acc.submit(inputs[0])             # shape (1, out)
            else:
                # Multiple inputs in one request — submit each separately so
                # they can be batched with other concurrent requests
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    futures = [pool.submit(acc.submit, inputs[i]) for i in range(len(inputs))]
                    results = [f.result() for f in futures]
                result = np.concatenate(results, axis=0)

        except queue.Full:
            _metrics.record(0, error=True)
            raise HTTPException(status_code=503, detail="Server overloaded — queue full")

        latency_ms = (time.perf_counter() - t_start) * 1000
        _metrics.record(latency_ms, batch_size=inputs.shape[0])

        return PredictResponse(
            predictions=result.tolist(),
            latency_ms=round(latency_ms, 3),
            batch_size=inputs.shape[0],
        )

    @app.post("/batch_predict", response_model=PredictResponse)
    async def batch_predict(req: PredictRequest):
        """
        Explicit batch inference — the whole batch is processed together.
        Use this when you control the batch yourself and don't want dynamic batching.
        """
        t_start = time.perf_counter()
        m = _get_model()

        inputs = np.array(req.inputs, dtype=np.float32)
        result = m.predict(inputs)

        latency_ms = (time.perf_counter() - t_start) * 1000
        _metrics.record(latency_ms, batch_size=inputs.shape[0])

        return PredictResponse(
            predictions=result.tolist(),
            latency_ms=round(latency_ms, 3),
            batch_size=inputs.shape[0],
        )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        m   = _get_model()
        acc = _get_accumulator()
        return HealthResponse(
            status="healthy",
            model=repr(m),
            requests_served=_metrics.requests_total,
            uptime_seconds=round(_metrics.uptime_seconds(), 1),
            queue_size=acc.queue_size(),
            throughput_rps=round(_metrics.throughput_rps(), 2),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        """Prometheus-format metrics."""
        return PlainTextResponse(
            content=_metrics.prometheus_text(),
            media_type="text/plain",
        )

    @app.get("/")
    async def root():
        return {"message": "ML Inference Server", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HAS_FASTAPI:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        sys.exit(1)

    import uvicorn
    print("Starting ML Inference Server on http://0.0.0.0:8000")
    print("API docs: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
