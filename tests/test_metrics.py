"""Tests for metrics and observability features."""

import pytest

from qmcp.metrics import Histogram, MetricsRegistry


# Note: The 'client' fixture is provided by conftest.py with proper DB isolation


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_endpoint_returns_text(self, client):
        """Metrics endpoint returns plain text."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_json_endpoint(self, client):
        """Metrics JSON endpoint returns stats."""
        response = client.get("/metrics/json")
        assert response.status_code == 200
        data = response.json()
        assert "counters" in data
        assert "gauges" in data
        assert "histograms" in data

    def test_metrics_include_request_counts(self, client):
        """Making requests increments counters."""
        # Make a request first
        client.get("/health")

        # Check metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        # Should have HTTP request metrics
        assert "qmcp_http_requests_total" in content

    def test_tool_invocation_records_metrics(self, client):
        """Tool invocations are recorded in metrics."""
        # Invoke a tool
        client.post("/v1/tools/echo", json={"input": {"message": "test"}})

        # Check metrics
        response = client.get("/metrics")
        content = response.text
        assert "qmcp_tool_invocations_total" in content
        assert "qmcp_tool_duration_seconds" in content


class TestHistogram:
    """Tests for the Histogram class."""

    def test_observe_single_value(self):
        """Histogram records observations."""
        hist = Histogram()
        hist.observe(0.05)
        assert hist.count == 1
        assert hist.sum == 0.05

    def test_observe_multiple_values(self):
        """Histogram accumulates observations."""
        hist = Histogram()
        hist.observe(0.01)
        hist.observe(0.02)
        hist.observe(0.03)
        assert hist.count == 3
        assert hist.sum == pytest.approx(0.06)

    def test_bucket_counts(self):
        """Histogram buckets are counted correctly."""
        hist = Histogram()
        hist.observe(0.001)  # <= 0.005 bucket
        hist.observe(0.008)  # <= 0.01 bucket
        hist.observe(0.02)   # <= 0.025 bucket

        # Each value increments all buckets it fits into
        assert hist.counts[0.005] == 1  # Only 0.001 fits
        # 0.008 doesn't fit in 0.005, but fits in 0.01 and above
        assert hist.count == 3

    def test_prometheus_format(self):
        """Histogram generates Prometheus format."""
        hist = Histogram()
        hist.observe(0.01)
        output = hist.to_prometheus("test_histogram")
        assert "test_histogram_bucket" in output
        assert "test_histogram_sum" in output
        assert "test_histogram_count" in output


class TestMetricsRegistry:
    """Tests for the MetricsRegistry class."""

    def test_counter_increment(self):
        """Counter increments correctly."""
        registry = MetricsRegistry()
        registry.inc_counter("test_counter")
        registry.inc_counter("test_counter")

        stats = registry.get_stats()
        assert stats["counters"]["test_counter"][""] == 2

    def test_counter_with_labels(self):
        """Counter tracks labeled values separately."""
        registry = MetricsRegistry()
        registry.inc_counter("test_counter", {"method": "GET"})
        registry.inc_counter("test_counter", {"method": "POST"})
        registry.inc_counter("test_counter", {"method": "GET"})

        stats = registry.get_stats()
        assert 'method="GET"' in str(stats["counters"]["test_counter"])

    def test_gauge_set(self):
        """Gauge sets value correctly."""
        registry = MetricsRegistry()
        registry.set_gauge("test_gauge", 42.5)

        stats = registry.get_stats()
        assert stats["gauges"]["test_gauge"][""] == 42.5

    def test_histogram_observation(self):
        """Registry records histogram observations."""
        registry = MetricsRegistry()
        registry.observe_histogram("test_hist", 0.1)
        registry.observe_histogram("test_hist", 0.2)

        stats = registry.get_stats()
        assert stats["histograms"]["test_hist"][""]["count"] == 2
        assert stats["histograms"]["test_hist"][""]["sum"] == pytest.approx(0.3)

    def test_prometheus_output(self):
        """Registry generates Prometheus format."""
        registry = MetricsRegistry()
        registry.inc_counter("requests_total", {"path": "/test"})
        registry.set_gauge("active_connections", 5)
        registry.observe_histogram("request_duration", 0.1)

        output = registry.to_prometheus()
        assert "# TYPE requests_total counter" in output
        assert "# TYPE active_connections gauge" in output
        assert "# TYPE request_duration histogram" in output


class TestRequestTracing:
    """Tests for request tracing middleware."""

    def test_request_id_header(self, client):
        """Responses include X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_correlation_id_header(self, client):
        """Responses include X-Correlation-ID header."""
        response = client.get("/health")
        assert "X-Correlation-ID" in response.headers

    def test_correlation_id_preserved(self, client):
        """Correlation ID from request is preserved in response."""
        response = client.get(
            "/health",
            headers={"X-Correlation-ID": "test-correlation-123"}
        )
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"

    def test_new_correlation_id_generated(self, client):
        """New correlation ID generated if not provided."""
        response = client.get("/health")
        correlation_id = response.headers["X-Correlation-ID"]
        assert correlation_id is not None
        assert len(correlation_id) > 0
