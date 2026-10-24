"""Dedicated latency and throughput benchmarking script for TicketLens models."""

import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np
from ticketlens.config import load_config
from ticketlens.predict import TicketPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.benchmark")

TEST_PROMPTS = [
    "VPN dropped unexpectedly with TLS handshake timeout.",
    "Requesting second 27 inch monitor for desk setup.",
    "Root cause investigation: database connection pool exhaustion.",
    "Scheduled firewall rule modification for staging environment subnet.",
    "Outlook client fails to sync mailbox with exchange server.",
    "User cannot access shared network drive after password reset.",
    "Kubernetes node memory usage spiking to 95 percent on cluster.",
    "Need software license key for JetBrains PyCharm Pro.",
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark TicketLens model latency")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    args = parser.parse_args()

    predictor = TicketPredictor()
    available_models = predictor.get_loaded_models()

    if not available_models:
        print("Error: No models found. Please train models first.")
        return

    print("==================================================")
    print(f" TicketLens Latency Benchmark ({args.iterations} iterations per model)")
    print("==================================================")

    results = {}
    for m in available_models:
        # Warmup
        for p in TEST_PROMPTS:
            _ = predictor.predict(p, model_type=m)

        latencies = []
        for i in range(args.iterations):
            prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
            t0 = time.perf_counter()
            _ = predictor.predict(prompt, model_type=m)
            latencies.append((time.perf_counter() - t0) * 1000)

        lat = np.array(latencies)
        mean_lat = round(float(np.mean(lat)), 2)
        p50 = round(float(np.percentile(lat, 50)), 2)
        p95 = round(float(np.percentile(lat, 95)), 2)
        p99 = round(float(np.percentile(lat, 99)), 2)
        throughput = round(1000.0 / mean_lat, 2) if mean_lat > 0 else 0.0

        results[m] = {
            "mean_ms": mean_lat,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "throughput_req_per_sec": throughput,
        }

        print(f"\nModel: {m.upper()}")
        print(f"  P50 Latency:   {p50} ms")
        print(f"  P95 Latency:   {p95} ms")
        print(f"  P99 Latency:   {p99} ms")
        print(f"  Throughput:    {throughput} req/sec")

    cfg = load_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "benchmark_latency.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark metrics saved to: {reports_dir / 'benchmark_latency.json'}")


if __name__ == "__main__":
    main()
