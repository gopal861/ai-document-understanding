import json
import os
import threading
from typing import List


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

            # NEW: latency history (critical for P95 calculation)
            "latencies": []

        }

        self._load()


    def _load(self):

        if os.path.exists(_METRICS_PATH):

            try:

                with open(_METRICS_PATH, "r") as f:

                    data = json.load(f)

                    # Backward compatibility
                    if "latencies" not in data:
                        data["latencies"] = []

                    self._metrics = data

            except Exception:
                pass


    def _save(self):

        with open(_METRICS_PATH, "w") as f:

            json.dump(self._metrics, f, indent=2)


    def record_success(self, latency: float):

        with _lock:

            self._metrics["total_requests"] += 1

            self._metrics["successful_requests"] += 1

            self._metrics["total_latency"] += latency

            self._metrics["avg_latency"] = (
                self._metrics["total_latency"]
                / self._metrics["total_requests"]
            )

            # NEW: store individual latency
            self._metrics["latencies"].append(latency)

            self._save()


    def record_failure(self):

        with _lock:

            self._metrics["total_requests"] += 1

            self._metrics["failed_requests"] += 1

            self._save()


    def get_metrics(self):

        return self._metrics


    # NEW: compute percentile safely
    def get_latency_percentile(self, percentile: float) -> float:

        latencies: List[float] = self._metrics.get("latencies", [])

        if not latencies:
            return 0.0

        sorted_latencies = sorted(latencies)

        index = int(len(sorted_latencies) * percentile / 100)

        index = min(index, len(sorted_latencies) - 1)

        return sorted_latencies[index]


metrics_tracker = MetricsTracker()

