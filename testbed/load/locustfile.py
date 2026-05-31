from __future__ import annotations

import math
import os
import random

from locust import HttpUser, between, events, task


LOAD_PROFILE = os.environ.get("LOAD_PROFILE", "normal").lower()


def profile_weight(endpoint: str) -> int:
    weights = {
        "normal": {"index": 8, "cpu": 2, "io": 2, "error": 1},
        "gradual": {"index": 5, "cpu": 4, "io": 3, "error": 1},
        "spike": {"index": 2, "cpu": 8, "io": 4, "error": 3},
        "stress": {"index": 1, "cpu": 10, "io": 7, "error": 4},
        "recovery": {"index": 7, "cpu": 2, "io": 2, "error": 1},
    }
    return weights.get(LOAD_PROFILE, weights["normal"])[endpoint]


class WebCongestionUser(HttpUser):
    wait_time = between(0.05, 0.8)

    @task(profile_weight("index"))
    def index(self):
        self.client.get("/")

    @task(profile_weight("cpu"))
    def cpu(self):
        factor = {
            "normal": random.uniform(0.2, 1.0),
            "gradual": random.uniform(1.0, 4.0),
            "spike": random.uniform(4.0, 10.0),
            "stress": random.uniform(8.0, 18.0),
            "recovery": random.uniform(0.5, 2.0),
        }.get(LOAD_PROFILE, 1.0)
        self.client.get(f"/cpu?factor={factor:.3f}", name="/cpu")

    @task(profile_weight("io"))
    def io_wait(self):
        delay = {
            "normal": random.uniform(0.005, 0.04),
            "gradual": random.uniform(0.02, 0.15),
            "spike": random.uniform(0.10, 0.60),
            "stress": random.uniform(0.25, 1.25),
            "recovery": random.uniform(0.01, 0.08),
        }.get(LOAD_PROFILE, 0.05)
        self.client.get(f"/io?delay={delay:.3f}", name="/io")

    @task(profile_weight("error"))
    def maybe_error(self):
        p = {
            "normal": 0.005,
            "gradual": 0.02,
            "spike": 0.08,
            "stress": 0.15,
            "recovery": 0.01,
        }.get(LOAD_PROFILE, 0.01)
        self.client.get(f"/maybe-error?p={p:.3f}", name="/maybe-error")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"LOAD_PROFILE={LOAD_PROFILE}")

