---
name: platform-reliability-management
description: >-
  Platform reliability management skill covering SRE operations, Kubernetes cluster
  health, ingress configurations, database latency profiling, message queue backlog,
  Redis/Kafka scaling, resource fragmentation, and deployment rollout safety.
---

# Platform Reliability Management (SRE)

This skill document defines SRE operational workflows, runtime metrics monitoring, queue diagnostics, and cluster scaling rules for the **Insterest** project.

---

## 1. Kubernetes & Runtime Resource Auditing

Maintaining cluster health requires monitoring CPU scheduling, memory limits, and container cycles.

### A. Diagnosing Memory Pressure & Fragmentation
- **Exit Code 137 (OOMKilled)**: Indicates a pod's physical memory footprint exceeded its manifest `limits.memory` cap.
- **Diagnostics**:
  1. Inspect limits: `kubectl describe pod <pod_name>`
  2. Profile service memory usage: Look for cached ML models (CLIP/SBERT) or open connection pools.
- **Autoscaling Configuration (HPA)**:
  Ensure `HorizontalPodAutoscaler` manifests (`infra/k8s/hpa.yaml` or equivalent) scale based on CPU utilization and memory thresholds:
  ```yaml
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
  ```

### B. Ingress Routing & Network Latency Auditing
Verify timeouts inside `infra/k8s/ingress.yaml` to prevent gateway timeouts (`504 Gateway Timeout`) on long-running ML jobs or file downloads:
*   `nginx.ingress.kubernetes.io/proxy-read-timeout: "180"`
*   `nginx.ingress.kubernetes.io/proxy-send-timeout: "180"`

---

## 2. Distributed Runtime & Middleware Health

In a microservices environment, system failures are often caused by interactions between database brokers, message queues, and worker pods.

### A. Kafka Consumer Group & Backlog Audit
Consumer lag can delay event-driven synchronization (e.g. upload-service file metadata syncing to recommendation-service).
- **Command Checklist**:
  ```bash
  # Check active Kafka consumer group statuses and lags
  kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group upload-service-group-v6
  ```
- **Lag Remediation**:
  1. If lag is increasing continuously: Check if the consumer loop is blocked by synchronous CPU/file operations (e.g., trimming files on the event loop).
  2. Increase replicas up to the partition count to run concurrent consumer processes.

### B. Redis Health & Cache Performance
Redis stores HNSW vector indices and interaction caches. Audit for memory saturation or high latency:
*   **Monitor Latency**: Run `redis-cli --latency -h <host>` to check network delay.
*   **Analyze Memory Saturation**: Execute `INFO memory` in Redis CLI to check `used_memory_human` and `mem_fragmentation_ratio`.
*   **LRU/Eviction Audit**: Check if eviction policies are configured correctly (e.g., `volatile-lru` or `allkeys-lru`) to prevent Redis from throwing `OOM command not allowed when used memory > 'maxmemory'`.
