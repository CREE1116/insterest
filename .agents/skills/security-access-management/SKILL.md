---
name: security-access-management
description: >-
  Security and access management skill covering authentication, authorization (RBAC),
  secret rotation, JWT lifecycle, CORS/CSRF, SQLi/SSRF prevention, rate limiting,
  and OWASP security guidelines.
---

# Security & Access Management

This skill document defines security standards, token lifecycles, access control models, and validation routines to protect Insterest microservices and infrastructure.

---

## 1. Authentication & JWT Lifecycle

Secure session management is critical for user data privacy.

### A. JWT Configuration Rules
*   **Signature Verification**: Always verify signatures using strong hashing algorithms (e.g. `HS256` or `RS256`).
*   **Expiration Constraints**: Set short lifespans for Access Tokens (e.g., 15-30 minutes) and use Refresh Tokens stored in secure, HttpOnly, SameSite=Strict cookies to obtain new access tokens.
*   **Token Expiry Audit Checklist**:
    ```python
    # Ensure JWT expiration is strictly verified on decode
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"], options={"require": ["exp"]})
    ```

### B. Secret Management & Credential Leakage
*   **No Hardcoded Secrets**: Never commit private keys, passwords, API tokens, or secrets to Git.
*   **Kubernetes Secrets**: Inject secrets via environment variables linked to Kubernetes Secret configurations:
    ```yaml
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: password
    ```
*   Run Git hooks or static scanning (e.g. `gitleaks`) before pushing to verify no credentials are leakable.

---

## 2. Web Vulnerability Auditing (OWASP Top 10)

Microservice APIs must sanitize input and restrict network routing.

### A. SQL Injection (SQLi)
*   Always use Object-Relational Mappers (SQLAlchemy AsyncSession) or parameterized queries.
*   **Never construct raw SQL strings** using variable concatenation (`f"SELECT * FROM users WHERE name = '{input}'"`).
*   **Correct Pattern**:
    ```python
    from sqlalchemy.future import select
    query = select(User).where(User.username == input_username)
    ```

### B. CSRF, CORS & SSRF Prevention
*   **CORS (Cross-Origin Resource Sharing)**: Restrict Allowed Origins in FastAPI middlewares to trusted domains only. Do not use wildcard `"*"` for services returning user-specific resources or credentials.
    ```python
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://insterest.app"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    ```
*   **SSRF (Server-Side Request Forgery)**: Sanitize and validate external links before fetching contents. Do not allow resolving to internal loopback addresses (e.g., `127.0.0.1`, `10.x.x.x`, `169.254.169.254` AWS metadata).

---

## 3. Rate Limiting & Access Control (RBAC)

Ensure users and API clients cannot exhaust resources or access unauthorized scopes.

*   **Role-Based Access Control (RBAC)**: Enforce route dependencies checking user scopes/roles (e.g., `current_user.is_admin`) before allowing execution of write or configuration actions.
*   **Rate Limiting**: Enforce API rate-limiting rules (e.g. using Redis-backed token buckets) on public endpoints like `/api/v1/auth/login` to prevent brute-force credential stuffing.
