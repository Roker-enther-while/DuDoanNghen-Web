from __future__ import annotations

import math
import os
import random
import time

from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


app = Flask(__name__)

REQUESTS = Counter("webapp_requests_total", "Total HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "webapp_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INFLIGHT = Gauge("webapp_inflight_requests", "In-flight HTTP requests")
ERRORS = Counter("webapp_errors_total", "Total simulated application errors", ["endpoint"])
WORK_FACTOR = int(os.environ.get("APP_WORK_FACTOR", "25000"))


def busy_work(multiplier: float) -> float:
    loops = max(1, int(WORK_FACTOR * multiplier))
    acc = 0.0
    for i in range(loops):
        acc += math.sin(i % 97) * math.cos(i % 31)
    return acc


def record(endpoint: str, started: float, status: int) -> None:
    LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - started)
    REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


@app.before_request
def before_request() -> None:
    INFLIGHT.inc()


@app.after_request
def after_request(response):
    INFLIGHT.dec()
    return response


@app.get("/")
def index():
    started = time.perf_counter()
    busy_work(0.2)
    status = 200
    record("/", started, status)
    return jsonify({"service": "congestion-webapp", "status": "ok"})


@app.get("/cpu")
def cpu():
    started = time.perf_counter()
    factor = float(request.args.get("factor", "1.0"))
    busy_work(max(0.1, min(factor, 20.0)))
    status = 200
    record("/cpu", started, status)
    return jsonify({"work": factor})


@app.get("/io")
def io_wait():
    started = time.perf_counter()
    delay = max(0.0, min(float(request.args.get("delay", "0.05")), 2.0))
    time.sleep(delay)
    status = 200
    record("/io", started, status)
    return jsonify({"delay": delay})


@app.get("/maybe-error")
def maybe_error():
    started = time.perf_counter()
    p = max(0.0, min(float(request.args.get("p", "0.02")), 1.0))
    if random.random() < p:
        status = 503
        ERRORS.labels(endpoint="/maybe-error").inc()
        record("/maybe-error", started, status)
        return jsonify({"error": "simulated overload"}), status
    busy_work(0.4)
    status = 200
    record("/maybe-error", started, status)
    return jsonify({"status": "ok", "p": p})


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)

