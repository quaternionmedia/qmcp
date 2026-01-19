"""Application metrics for observability.

Provides Prometheus-compatible metrics for:
- HTTP request counts and latencies
- Tool invocation counts and latencies
- HITL request counts and response times
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class Histogram:
    """Simple histogram for latency tracking."""

    buckets: list[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    counts: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    sum: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket:
                self.counts[bucket] += 1

    def to_prometheus(self, name: str, labels: str = "") -> str:
        """Generate Prometheus histogram format."""
        lines = []
        label_str = f"{{{labels}}}" if labels else ""

        # Bucket lines (counts are already cumulative in observe())
        for bucket in self.buckets:
            lines.append(
                f'{name}_bucket{{le="{bucket}"{", " + labels if labels else ""}}} '
                f"{self.counts[bucket]}"
            )
        lines.append(f'{name}_bucket{{le="+Inf"{", " + labels if labels else ""}}} {self.count}')

        # Sum and count
        lines.append(f"{name}_sum{label_str} {self.sum}")
        lines.append(f"{name}_count{label_str} {self.count}")

        return "\n".join(lines)


class MetricsRegistry:
    """Simple metrics registry for application metrics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._gauges: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: dict[str, dict[str, Histogram]] = defaultdict(dict)

    def inc_counter(self, name: str, labels: dict[str, str] | None = None, value: int = 1) -> None:
        """Increment a counter metric."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._counters[name][label_key] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][label_key] = value

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            if label_key not in self._histograms[name]:
                self._histograms[name][label_key] = Histogram()
            self._histograms[name][label_key].observe(value)

    def _labels_to_key(self, labels: dict[str, str] | None) -> str:
        """Convert labels dict to a hashable key."""
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def to_prometheus(self) -> str:
        """Generate Prometheus text format output."""
        lines = []

        with self._lock:
            # Counters
            for name, label_values in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                for label_key, value in label_values.items():
                    label_str = f"{{{label_key}}}" if label_key else ""
                    lines.append(f"{name}{label_str} {value}")
                lines.append("")

            # Gauges
            for name, label_values in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                for label_key, value in label_values.items():
                    label_str = f"{{{label_key}}}" if label_key else ""
                    lines.append(f"{name}{label_str} {value}")
                lines.append("")

            # Histograms
            for name, label_histograms in self._histograms.items():
                lines.append(f"# TYPE {name} histogram")
                for label_key, histogram in label_histograms.items():
                    lines.append(histogram.to_prometheus(name, label_key))
                lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """Get metrics as a dictionary (for JSON endpoints)."""
        with self._lock:
            return {
                "counters": {k: dict(v) for k, v in self._counters.items()},
                "gauges": {k: dict(v) for k, v in self._gauges.items()},
                "histograms": {
                    k: {lk: {"count": h.count, "sum": h.sum} for lk, h in v.items()}
                    for k, v in self._histograms.items()
                },
            }


# Global metrics registry
metrics = MetricsRegistry()


# Convenience functions
def record_request(method: str, path: str, status_code: int, duration: float) -> None:
    """Record an HTTP request."""
    labels = {"method": method, "path": path, "status": str(status_code)}
    metrics.inc_counter("qmcp_http_requests_total", labels)
    metrics.observe_histogram("qmcp_http_request_duration_seconds", duration, labels)


def record_tool_invocation(tool_name: str, status: str, duration: float) -> None:
    """Record a tool invocation."""
    labels = {"tool": tool_name, "status": status}
    metrics.inc_counter("qmcp_tool_invocations_total", labels)
    metrics.observe_histogram("qmcp_tool_duration_seconds", duration, {"tool": tool_name})


def record_hitl_request(request_type: str, action: str) -> None:
    """Record a HITL request action."""
    labels = {"type": request_type, "action": action}
    metrics.inc_counter("qmcp_hitl_requests_total", labels)
