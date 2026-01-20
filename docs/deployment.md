# Deployment Guide

This guide covers deploying QMCP in various environments.

## Quick Start (Development)

```bash
# Clone and install
git clone <repo-url>
cd qmcp
uv sync

# Start the server
uv run qmcp serve --reload
```

## Production Deployment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QMCP_HOST` | `127.0.0.1` | Server bind address |
| `QMCP_PORT` | `3333` | Server port |
| `QMCP_DEBUG` | `false` | Enable debug mode |
| `QMCP_DATABASE_URL` | `sqlite+aiosqlite:///./qmcp.db` | Database connection |

### Running with Uvicorn

```bash
# Production mode (JSON logging)
uvicorn qmcp.server:app --host 0.0.0.0 --port 3333 --workers 4

# With environment variables
QMCP_HOST=0.0.0.0 QMCP_PORT=3333 uv run qmcp serve
```

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY qmcp/ qmcp/

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 3333

# Run server
CMD ["uv", "run", "qmcp", "serve", "--host", "0.0.0.0"]
```

Build and run:

```bash
docker build -t qmcp .
docker run -p 3333:3333 qmcp
```

### Docker Compose

```yaml
version: '3.8'
services:
  qmcp:
    build: .
    ports:
      - "3333:3333"
    environment:
      - QMCP_HOST=0.0.0.0
      - QMCP_PORT=3333
      - QMCP_DEBUG=false
    volumes:
      - qmcp-data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qmcp-data:
```

## Observability

### Logging

QMCP uses structured logging via structlog:

- **Development**: Human-readable console output
- **Production**: JSON-formatted logs

Log output includes:
- `request_id` - Unique ID per request
- `correlation_id` - Tracking ID across services
- `method`, `path` - HTTP request details
- `duration_ms` - Request timing

Example JSON log:
```json
{
  "timestamp": "2026-01-18T12:00:00Z",
  "level": "info",
  "event": "request_completed",
  "request_id": "abc123",
  "correlation_id": "xyz789",
  "method": "POST",
  "path": "/v1/tools/echo",
  "status_code": 200,
  "duration_ms": 15.2
}
```

### Auditability and Accountability

QMCP maintains durable audit records in SQLite for:

| Record | Where | Why it matters |
|--------|-------|----------------|
| Tool invocation | `tool_invocations` | Full input/output, status, duration, and timestamps for every tool call |
| Human request | `human_requests` | Request prompt, options, timeouts, and lifecycle status |
| Human response | `human_responses` | Response payload, `responded_by`, and response metadata |

Operational accountability is strengthened by:
- `correlation_id` passed by clients and stored with tool invocations and human requests
- `request_id` + `correlation_id` in logs and response headers for end-to-end traceability
- `responded_by` and `response_metadata` on human responses to preserve decision provenance

**Recommended practice:** always pass `X-Correlation-ID` from callers and store any human approver identity in `responded_by`.

### Metrics

Prometheus-compatible metrics are available at `/metrics`:

```bash
curl http://localhost:3333/metrics
```

**Available metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `qmcp_http_requests_total` | Counter | Total HTTP requests |
| `qmcp_http_request_duration_seconds` | Histogram | Request latency |
| `qmcp_tool_invocations_total` | Counter | Tool invocations |
| `qmcp_tool_duration_seconds` | Histogram | Tool execution time |
| `qmcp_hitl_requests_total` | Counter | HITL requests |

**Prometheus scrape config:**

```yaml
scrape_configs:
  - job_name: 'qmcp'
    static_configs:
      - targets: ['qmcp:3333']
    metrics_path: '/metrics'
```

### Health Checks

```bash
# Simple health check
curl http://localhost:3333/health

# Response
{"status": "healthy", "version": "0.1.0"}
```

## Request Tracing

QMCP automatically adds tracing headers:

**Request headers (optional):**
- `X-Correlation-ID` - Pass your own correlation ID

**Response headers (always present):**
- `X-Request-ID` - Unique request identifier
- `X-Correlation-ID` - Correlation ID (yours or generated)

Use correlation IDs to trace requests across services:

```python
import httpx

response = httpx.post(
    "http://localhost:3333/v1/tools/echo",
    json={"input": {"message": "hello"}},
    headers={"X-Correlation-ID": "my-workflow-123"}
)
print(response.headers["X-Request-ID"])  # e.g., "abc12345"
print(response.headers["X-Correlation-ID"])  # "my-workflow-123"
```

## Database

### SQLite (Default)

The default SQLite database works well for:
- Development
- Single-instance deployments
- Low-to-medium traffic

Database file: `./qmcp.db`

### Production Considerations

For high-availability deployments, consider:
- PostgreSQL with `asyncpg` driver
- Shared storage for SQLite
- Database backups

## Security Considerations

### Network Security

- Bind to `127.0.0.1` unless external access needed
- Use reverse proxy (nginx, Traefik) for TLS
- Implement rate limiting at proxy layer

### Future Enhancements

Authentication and authorization are planned for future releases:
- API key authentication
- OAuth2/OIDC integration
- Role-based access control

## Troubleshooting

### Common Issues

**Database locked:**
```
sqlite3.OperationalError: database is locked
```
- Ensure only one instance writes to SQLite
- Consider PostgreSQL for concurrent access

**Port in use:**
```
OSError: [Errno 98] Address already in use
```
- Change port with `--port` flag
- Kill existing process: `lsof -i :3333`

### Debug Mode

Enable debug mode for verbose output:

```bash
QMCP_DEBUG=true uv run qmcp serve
```

This enables:
- Console-formatted logs (not JSON)
- DEBUG level logging
- FastAPI debug mode

## Monitoring Checklist

- [ ] Health endpoint accessible
- [ ] Metrics scraped by Prometheus
- [ ] Logs shipped to aggregator
- [ ] Alerts configured for errors
- [ ] Database backups scheduled

## Local Implementation Test Phase

Use this phase to validate a fresh local install before deploying:

1. **Install and verify config**
   - `uv sync`
   - `uv run qmcp info`
2. **Start server**
   - `uv run qmcp serve`
3. **API smoke checks**
   - `GET /health`
   - `GET /v1/tools`
   - `POST /v1/tools/echo`
4. **Audit trail checks**
   - `GET /v1/invocations` (verify new invocation is present)
   - Create/response HITL request and verify status transitions
5. **Observability checks**
   - `GET /metrics` and `GET /metrics/json`
   - Verify `X-Request-ID` and `X-Correlation-ID` headers on responses

## QC Gauntlet

Repeat this gate before releases:
- [ ] All tests pass (`uv run qmcp test`)
- [ ] Restart server and confirm prior invocations and HITL records remain
- [ ] Logs show `request_id` and `correlation_id` for a multi-request flow
- [ ] Metrics scrape is clean; request and tool counters increment
- [ ] HITL response audit shows `responded_by` populated
