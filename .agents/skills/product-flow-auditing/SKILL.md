---
name: product-flow-auditing
description: >-
  Product flow auditing skill covering UX-backend alignment, frontend-backend contract
  consistency, pagination ergonomics, infinite scroll race conditions, and optimistic updates.
---

# Product Flow Auditing

This skill document defines guidelines to audit frontend-backend interface consistency, user onboarding flows, and asynchronous client-state synchronization.

---

## 1. Frontend-Backend Contract Consistency

Ensure API schemas (defined via Pydantic/FastAPI) match the types, fields, and expectations defined in the frontend typescript configurations.

### A. API Ergonomics & Contract Validation
*   Check that JSON responses do not return fields with nested structures that the frontend does not parse.
*   Ensure date and datetime strings are formatted in ISO 8601 UTC format (`YYYY-MM-DDTHH:MM:SSZ`) to avoid client-side timezone parsing errors.
*   Ensure status updates or error models return explicit error codes (`error_code`) rather than generic string messages, allowing the frontend to show localized error blocks or custom dialogs.

---

## 2. Pagination & Infinite Scroll Architecture

Web service performance depends on correct pagination models.

### A. Cursor-Based Pagination Rules
For high-frequency feeds (such as recommendation posts or interactions), use **Cursor-Based Pagination** instead of Offset/Limit to prevent duplicate items or missed items when new posts are created:
*   **Request Schema**: `/api/v1/posts?cursor=<encoded_timestamp_id>&limit=20`
*   **Response Schema**:
    ```json
    {
      "items": [...],
      "next_cursor": "eyJjcmVhdGVkX2F0IjogIjIwMjYtMDUtMjJUMDU6NTg6MjJaIiwgImlkIjogNDJ9",
      "has_more": true
    }
    ```
*   **Cursor Design**: Enforce base64 encoding of structured data (e.g. `(created_at, id)`) to keep query lookups efficient and database indices utilized.

### B. Infinite Scroll Race Conditions
*   Verify that backend queries handle pagination limits gracefully, returning an empty list `[]` and `has_more: false` once bounds are reached, rather than returning `404 Not Found` or looping.
*   Ensure response payloads include consistent identifiers to allow the frontend to deduplicate components on incoming stream updates.

---

## 3. Empty States & Optimistic Update Sync

Ensure graceful UI fallbacks and state coherence when operations fail asynchronously.

### A. Graceful Empty States & Fallbacks
*   **Recommendation Empty State**: If ANN retrieval fails, return a curated, static fallback feed (popular posts) rather than throwing `500 Internal Server Error` or rendering a blank container.
*   **Search Empty State**: Standardize response formats for zero results, distinguishing between network timeouts and actual empty matches.

### B. Optimistic Update Drift Mitigation
When users interact with UI elements (e.g. liking a post or following an account), the frontend immediately updates state locally (optimistic update) before the backend API confirms success.
*   **Audit Rules**:
    1.  If the API call fails, the client MUST roll back the optimistic UI state.
    2.  Check that API error responses are clear enough (e.g. returning `409 Conflict` or explicit status codes) to trigger client-side rollback mechanisms and alert the user.
