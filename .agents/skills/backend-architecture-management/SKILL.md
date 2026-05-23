---
name: backend-architecture-management
description: >-
  Backend architecture management skill covering API design integrity, domain modeling,
  database transaction boundaries, eventual consistency guarantees, message queue
  semantics, retry deduplication, and service idempotency.
---

# Backend Architecture Management

This skill document defines guidelines for backend design, transactional safety, database session management, and asynchronous event consistency across microservices.

---

## 1. Domain Modeling & Transaction Boundaries

To maintain data integrity in distributed environments, every database write operation must occur within a well-defined transaction boundary.

### A. Database Session Lifecycles
*   **Prevent Connection Starvation**: Ensure connection objects or DB sessions are yielded, used, and closed immediately. Do not keep transactions open during slow external network calls (e.g. while querying Hugging Face or Pollinations APIs).
*   **FastAPI get_db Pattern**:
    *   Do not run commits within the dependency yielding block. Commits should be performed explicitly inside service handlers or endpoints:
    ```python
    async def create_user_item(db: AsyncSession, item_data):
        db.add(item_data)
        await db.commit() # Localized commit within the function scope
        await db.refresh(item_data)
    ```

### B. Service Boundaries & API Spec Integrity
*   Ensure HTTP status codes match REST conventions:
    *   `201 Created` for successfully generated resources.
    *   `400 Bad Request` for model validation failures.
    *   `422 Unprocessable Entity` for request model validation errors.
*   Cross-examine endpoints against `API_SPEC.md` to prevent duplicate routing prefixes or conflicting path structures (e.g. ingress mapping disputes).

---

## 2. Event Semantics & Eventual Consistency

Microservices use Kafka to publish and consume events asynchronously (e.g., `post-created`, `generation-completed`). This creates eventual consistency delays.

### A. Idempotency & Retry Deduplication
Network partitions and container restarts can cause Kafka to deliver duplicate messages (At-Least-Once delivery).
*   **Idempotency Keys**: Ensure message consumption is idempotent by checking unique message keys (such as `content_id` or `uuid`) against database tables before executing inserts:
    ```python
    # Check if this generation meta is already persisted
    exists = await db.execute(select(GenerationMeta).where(GenerationMeta.content_id == content_id))
    if exists.scalar():
        logger.info(f"Duplicate event ignored for content: {content_id}")
        return
    ```

### B. Eventual Consistency Handling
When a user likes a post, the transaction completes in the `interaction-service`, but the `recommendation-service` updating user taste vectors runs asynchronously.
*   **Mitigation**:
    1. Provide immediate UX feedback (optimistic update on the frontend).
    2. Maintain graceful fallback states in recommendation search (e.g., falling back to text-only retrieval if user taste vectors are currently being computed).
