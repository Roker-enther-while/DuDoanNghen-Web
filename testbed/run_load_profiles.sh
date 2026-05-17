#!/usr/bin/env bash
set -euo pipefail

HOST_URL="${HOST_URL:-http://localhost:8080}"
NORMAL_USERS="${NORMAL_USERS:-20}"
STRESS_USERS="${STRESS_USERS:-120}"
RUN_TIME="${RUN_TIME:-5m}"

run_profile() {
  local name="$1"
  local users="$2"
  local spawn="$3"
  echo "Running Locust profile ${name}"
  LOAD_PROFILE="$name" locust -f testbed/load/locustfile.py --headless \
    --host "$HOST_URL" \
    --users "$users" \
    --spawn-rate "$spawn" \
    --run-time "$RUN_TIME" \
    --csv "paper_artifacts/locust_${name}"
}

run_profile normal "$NORMAL_USERS" 5
run_profile gradual $(((NORMAL_USERS + STRESS_USERS) / 2)) 3
run_profile spike "$STRESS_USERS" 50
run_profile stress "$STRESS_USERS" 10
run_profile recovery "$NORMAL_USERS" 10

