"""
batching.py — Dynamic batching logic for the inference server.

BatchAccumulator collects requests from multiple clients, waits up to
max_wait_ms milliseconds OR until max_batch_size requests arrive, then
processes them all at once.  This increases GPU/CPU utilization by
amortizing the fixed cost of a forward pass over many inputs.
"""

import threading
import time
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np


@dataclass
class PendingRequest:
    """A single inference request waiting in the batch queue."""
    request_id: int
    input_data: np.ndarray                     # shape: (1, in_features)
    result_future: "threading.Event"           # set when result is ready
    result: Optional[np.ndarray] = None
    enqueue_time: float = field(default_factory=time.perf_counter)


class BatchMetrics:
    """Thread-safe metrics for the batch accumulator."""
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests   = 0
        self.total_batches    = 0
        self.total_batch_size = 0           # sum for computing mean
        self.batch_size_hist  = {}          # batch_size → count
        self.wait_times_ms    = []          # time request spent waiting

    def record_batch(self, batch_size: int):
        with self._lock:
            self.total_batches    += 1
            self.total_batch_size += batch_size
            self.batch_size_hist[batch_size] = \
                self.batch_size_hist.get(batch_size, 0) + 1

    def record_request(self, wait_ms: float):
        with self._lock:
            self.total_requests += 1
            self.wait_times_ms.append(wait_ms)

    def mean_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_batch_size / self.total_batches

    def batch_size_distribution(self) -> dict:
        with self._lock:
            return dict(self.batch_size_hist)


class BatchAccumulator:
    """
    Collects inference requests and dispatches them in batches.

    The worker thread drains the queue when EITHER:
      - max_batch_size requests have accumulated, OR
      - max_wait_ms milliseconds have elapsed since the first request arrived.

    This implements the classic "bounded wait" dynamic batching strategy
    used in TensorFlow Serving, Triton Inference Server, etc.

    Parameters
    ----------
    model_fn      : callable (np.ndarray) → np.ndarray
                    Takes a batched input (N, in_features), returns (N, out)
    max_batch_size: int    — flush when this many requests accumulate
    max_wait_ms   : float  — flush after this many ms even if batch is small
    max_queue_size: int    — return 503 if queue exceeds this
    """

    def __init__(self,
                 model_fn: Callable,
                 max_batch_size: int = 32,
                 max_wait_ms:    float = 10.0,
                 max_queue_size: int = 256):
        self.model_fn       = model_fn
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms
        self.max_queue_size = max_queue_size
        self.metrics        = BatchMetrics()

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._request_id_counter = 0
        self._counter_lock       = threading.Lock()
        self._stop_event         = threading.Event()

        # Start the background worker
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="batch-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def submit(self, input_data: np.ndarray) -> np.ndarray:
        """
        Submit a single input for inference.  Blocks until result is ready.

        input_data : shape (in_features,) or (1, in_features)
        Returns    : output np.ndarray
        Raises     : queue.Full if the server is overloaded
        """
        if input_data.ndim == 1:
            input_data = input_data[np.newaxis, :]

        with self._counter_lock:
            req_id = self._request_id_counter
            self._request_id_counter += 1

        event  = threading.Event()
        req    = PendingRequest(
            request_id=req_id,
            input_data=input_data,
            result_future=event,
        )

        # Will raise queue.Full if server is overloaded (caller catches it)
        self._queue.put_nowait(req)
        event.wait()    # Block until worker sets the result

        wait_ms = (time.perf_counter() - req.enqueue_time) * 1000
        self.metrics.record_request(wait_ms)

        return req.result

    def _worker_loop(self):
        """
        Background thread: drain the queue in batches.
        Uses a timed-wait pattern to respect max_wait_ms.
        """
        while not self._stop_event.is_set():
            # Wait for at least one request
            try:
                first_req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Accumulate more requests up to max_batch_size / max_wait_ms
            batch = [first_req]
            deadline = time.perf_counter() + self.max_wait_ms / 1000.0

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                    batch.append(req)
                except queue.Empty:
                    break

            self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: list):
        """
        Run model inference on a batch of requests and deliver results.
        """
        self.metrics.record_batch(len(batch))

        # Stack inputs into a single array
        batched_input = np.concatenate([r.input_data for r in batch], axis=0)

        try:
            batched_output = self.model_fn(batched_input)
        except Exception as e:
            # On error, set None result and unblock all waiters
            for req in batch:
                req.result = None
                req.result_future.set()
            return

        # Distribute results
        for i, req in enumerate(batch):
            req.result = batched_output[i:i+1]
            req.result_future.set()

    def stop(self):
        self._stop_event.set()
        self._worker_thread.join(timeout=2.0)

    def queue_size(self) -> int:
        return self._queue.qsize()

    def is_overloaded(self) -> bool:
        return self._queue.qsize() >= self.max_queue_size * 0.9


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import concurrent.futures

    def mock_model(x: np.ndarray) -> np.ndarray:
        time.sleep(0.005)   # simulate 5ms inference
        return x @ np.ones((x.shape[1], 10), dtype=np.float32)

    acc = BatchAccumulator(mock_model, max_batch_size=16, max_wait_ms=10.0)

    print("Sending 50 concurrent requests...")
    inputs = [np.random.randn(64).astype(np.float32) for _ in range(50)]

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(acc.submit, inp) for inp in inputs]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed*1000:.1f} ms")
    print(f"Mean batch size: {acc.metrics.mean_batch_size():.1f}")
    print(f"Total batches:   {acc.metrics.total_batches}")
    print(f"Batch size distribution: {acc.metrics.batch_size_distribution()}")
    print(f"All results shape: {[r.shape for r in results[:3]]}")
    acc.stop()
