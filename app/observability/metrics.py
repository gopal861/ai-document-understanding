import json
import os
import threading


_METRICS_PATH = "storage/metrics.json"

_lock = threading.Lock()


class MetricsTracker:

    def __init__(self):

        os.makedirs("storage", exist_ok=True)

        self._metrics = {

            "total_requests": 0,

            "successful_requests": 0,

            "failed_requests": 0,

            "total_latency": 0.0,

            "avg_latency": 0.0,

        }

        self._load()


    def _load(self):

        if os.path.exists(_METRICS_PATH):

            try:

                with open(_METRICS_PATH, "r") as f:

                    self._metrics = json.load(f)

            except Exception:

                pass


    def _save(self):

        with open(_METRICS_PATH, "w") as f:

            json.dump(self._metrics, f)


    def record_success(self, latency):

        with _lock:

            self._metrics["total_requests"] += 1

            self._metrics["successful_requests"] += 1

            self._metrics["total_latency"] += latency

            self._metrics["avg_latency"] = (

                self._metrics["total_latency"]

                / self._metrics["total_requests"]

            )

            self._save()


    def record_failure(self):

        with _lock:

            self._metrics["total_requests"] += 1

            self._metrics["failed_requests"] += 1

            self._save()


    def get_metrics(self):

        return self._metrics


metrics_tracker = MetricsTracker()
