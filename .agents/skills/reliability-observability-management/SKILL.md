---
name: reliability-observability-management
description: >-
  Reliability and observability management skill covering distributed tracing, metrics,
  SLOs, latency profiling, anomaly detection, incident response, and retry storm mitigation.
---

# Reliability & Observability Management

This skill document defines guidelines for distributed tracing, metrics instrumentation, Service Level Objectives (SLOs), alerting configurations, and diagnosing cascading system failures.

---

## 1. Distributed Tracing & OpenTelemetry (OTel)

Tracing allows cross-service tracking of requests (e.g., `frontend` -> `auth-service` -> `user-service` -> `database`).

### A. Instrumentation Standards
*   **Trace Propagation**: Ensure all services extract and inject headers using the W3C Trace Context standard (`traceparent`).
*   **Span Semantics**: Custom spans must contain key attributes like `db.system: "postgresql"`, `http.status_code: 200`, or custom ML metadata like `ml.model_name: "fused-clip-128d"`.
*   **Async/Event Loop Safety**: Ensure tracing spans do not block Python’s asyncio event loop. Wrap tracing context managers appropriately:
    ```python
    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)

    async def fetch_user_profile(user_id: int):
        with tracer.start_as_current_span("fetch_user_profile") as span:
            span.set_attribute("user.id", user_id)
            # Call async DB query
            return await db.execute(...)
    ```

### B. Trace Visualisation & Bottleneck Analysis
*   Inspect distributed traces in Jaeger / Honeycomb.
*   Identify high-latency segments: Look for wide database spans (missing indexes) or sequential HTTP calls that could be parallelized with `asyncio.gather`.

---

## 2. Metrics & Service Level Objectives (SLOs)

We collect Prometheus-style metrics using custom client instrumentation to monitor service health.

### A. Core Metrics & Instrument Types
*   **Request Counter**: Tracks overall volume and HTTP response codes.
    ```python
    from prometheus_client import Counter
    REQUEST_COUNTER = Counter("http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"])
    ```
*   **Latency Histogram**: Profiles response latency percentiles (P95, P99).
    ```python
    from prometheus_client import Histogram
    LATENCY_HISTOGRAM = Histogram("http_request_duration_seconds", "HTTP Latency", ["endpoint"])
    ```

### B. Defining SLOs & Alerting Thresholds
Establish SLO targets to govern deployment safety and trigger alerts:
*   **Latency SLO**: $95\%$ of `/discover` recommendation requests must return in $< 100\text{ms}$.
*   **Availability SLO**: $99.9\%$ of API requests must return non-5xx status codes over a 7-day rolling window.
*   **PromQL Alerting Rules Example**:
    ```yaml
    alert: RecommendationLatencySpike
    expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "95th percentile recommendation latency exceeded 100ms"
    ```

---

## 3. Incident Diagnostics & Cascading Failures

Under heavy load, minor latencies can cascade into critical site-wide outages.

### A. The Retry Storm Cascade
A typical cascading pattern:
$$\text{Downstream Slowdown} \longrightarrow \text{Timeout} \longrightarrow \text{Aggressive Retries} \longrightarrow \text{Database/Redis Saturation}$$
*   **Mitigation**:
    1.  **Exponential Backoff with Jitter**: Avoid retrying immediately or at constant intervals. Add random noise (jitter) to distribute client requests over time:
        $$\text{Backoff} = \text{Base} \times 2^{\text{attempt}} + \text{Uniform}(0, \text{Jitter})$$
    2.  **Circuit Breaker**: Implement circuit breakers (e.g. using `tenacity` or custom middleware) to instantly fail fast when a downstream dependency (e.g., `recommendation-service`) is unhealthy, rather than saturating socket pools.

### B. Latency Heatmaps & Anomaly Detection
*   Analyze latency distribution shapes. A bimodal distribution indicates that a specific sub-population of requests (e.g., query search bypass vs full HNSW index retrieval) is experiencing extreme slow paths.
*   Identify pod restarts or CPU throttling spikes coinciding with latency degradation.
