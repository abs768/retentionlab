"""Instance-local metrics with a bounded endpoint label; never customer attributes."""

from prometheus_client import CollectorRegistry, Counter, Histogram

INFERENCE_PATHS = {"/predict", "/batch-predict", "/rank"}


class ServiceMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "retentionlab_inference_requests_total",
            "Inference HTTP requests",
            ["endpoint"],
            registry=self.registry,
        )
        self.errors = Counter(
            "retentionlab_inference_errors_total",
            "Failed inference requests",
            ["endpoint"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "retentionlab_predictions_total",
            "Customer probabilities returned",
            registry=self.registry,
        )
        self.latency = Histogram(
            "retentionlab_inference_latency_seconds",
            "Inference HTTP request duration",
            ["endpoint"],
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "retentionlab_batch_size",
            "Validated scoring batch sizes",
            buckets=(1, 10, 100, 500, 1000),
            registry=self.registry,
        )
        self.drift_checks = Counter(
            "retentionlab_drift_checks_total", "Completed drift checks", registry=self.registry
        )
