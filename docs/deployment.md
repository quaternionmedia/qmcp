# Deployment Guide

This guide covers deploying QMCP in various environments.

## Development

A server started to develop qmcp takes its own port and database:
[`contributing.md`](contributing.md), "A server of your own". The shared
install is in [`setup/README.md`](setup/README.md).

## Production Deployment

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QMCP_HOST` | `127.0.0.1` | Server bind address |
| `QMCP_PORT` | `3141` | Server port |
| `QMCP_DEBUG` | `false` | Enable debug mode |
| `QMCP_DATABASE_URL` | `sqlite+aiosqlite:///./qmcp.db` | Database connection |
| `QMCP_RUN_DIR` | `~/.qmcp/run` | Where `qmcp serve` leaves its run-file |
| `QMCP_SESSION` | unset | A label the server states on `/health` and in its run-file |

### Running the server

```bash
QMCP_HOST=0.0.0.0 uv run qmcp serve --port 3141
```

Run it through `qmcp serve`, one process: `serve` leaves the run-file a
collector on this machine finds, and refuses to start beside a healthy qmcp on
its port. `uvicorn qmcp.server:app` started directly serves the same app and
leaves no run-file. Bound off loopback, the server registers only the routes
[`http-api.md`](http-api.md) marks **any**, and whoever reaches the port can
invoke its tools and answer its human requests.

### Docker Deployment

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# The lock, the readme the package metadata names, and vox, a path dependency
COPY pyproject.toml uv.lock README.md ./
COPY vendor/vox ./vendor/vox
COPY qmcp ./qmcp

RUN uv sync --frozen --no-dev

EXPOSE 3141

# Off loopback inside the container, so the published port reaches it. The
# server asks nobody who they are: publish the port only where that is safe.
CMD ["uv", "run", "--no-sync", "qmcp", "serve", "--host", "0.0.0.0", "--port", "3141"]
```

Build and run, with the database on a volume and the port published to this
machine only:

```bash
docker build -t qmcp .
docker run -p 127.0.0.1:3141:3141 -v qmcp-data:/app/data   -e QMCP_DATABASE_URL=sqlite+aiosqlite:////app/data/qmcp.db qmcp
```

### Docker Compose

```yaml
services:
  qmcp:
    build: .
    ports:
      - "127.0.0.1:3141:3141"
    environment:
      - QMCP_DATABASE_URL=sqlite+aiosqlite:////app/data/qmcp.db
    volumes:
      - qmcp-data:/app/data
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3141/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qmcp-data:
```

The database is the human queue: on the volume it outlives the container, and
`qmcp db backup` copies it with the server running.

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
curl http://localhost:3141/metrics
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
      - targets: ['qmcp:3141']
    metrics_path: '/metrics'
```

### Health Checks

```bash
# Simple health check
curl http://localhost:3141/health

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
    "http://localhost:3141/v1/tools/echo",
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
- See what holds it: `lsof -i :3141` (Linux, macOS) or
  `Get-NetTCPConnection -LocalPort 3141` (Windows), and move yourself rather
  than stop a server you did not start

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
