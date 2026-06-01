# Inference Server

A production-ready ML inference server with FastAPI, dynamic batching,
and Prometheus metrics.

## Architecture

```
Client → POST /predict → BatchAccumulator → MLP model → Response
                         (waits up to 10ms,
                          groups N requests
                          into one batch)
```

## Running

```bash
pip install -r requirements.txt

# Start the server
python3 src/server.py

# In another terminal, test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs": [[0.1, 0.2, ...]]}'   # 64-dim input

# Load test (standalone — no server needed)
python3 src/load_test.py --standalone

# Load test against running server
python3 src/load_test.py --server http://localhost:8000
```

## Key Concepts

### Latency vs Throughput Tradeoff
- **Latency**: time for a single request to complete
- **Throughput**: requests processed per second
- With batching: throughput increases, but each request waits up to `max_wait_ms`
- Optimal batch size is hardware-dependent: GPUs are compute-bound at large
  batches; CPUs saturate earlier

### Dynamic Batching
The `BatchAccumulator` collects requests from multiple concurrent clients
and dispatches them together. This amortizes the fixed overhead of a model
forward pass over many inputs. A model that takes 5ms per request can serve
10x more requests per second if batched to size 10.

### Little's Law
Throughput = Concurrency / Latency (in consistent units):
```
λ = L / W
```
Where λ = throughput (req/s), L = mean concurrency, W = mean latency (s).
This fundamental law from queueing theory tells you: if you add more clients
(L), throughput only improves if the system can actually process faster —
otherwise you just inflate W (latency).

### Why p99 Matters More Than p50
The median (p50) latency looks good even when 1% of requests take 10x longer.
In a system with 100 downstream calls, the chance of at least one call hitting
p99 latency is `1 - 0.99^100 ≈ 63%`. So p99 latency is what users actually
experience in complex systems, not the median.

### Prometheus Metrics
- `inference_requests_total` — counter (always increases)
- `inference_latency_ms_p50/p95/p99` — percentile gauges
- `inference_mean_batch_size` — quality of batching
- `inference_throughput_rps` — overall health

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single input with dynamic batching |
| `/batch_predict` | POST | Explicit batch inference |
| `/health` | GET | Server status |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API docs (Swagger UI) |
