"""Latency benchmark for RxPredict inference pipeline.

Measures end-to-end latency including feature processing and model inference.
Verifies sub-100ms SLA compliance under various conditions.

Usage:
    python benchmarks/latency_benchmark.py --iterations 1000 --fail-threshold 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    """Benchmark result summary."""

    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float
    throughput_rps: float
    sla_pass: bool
    sla_threshold_ms: float


def generate_sample_patient(rng: np.random.RandomState) -> dict:
    """Generate a random patient record for benchmarking."""
    drug_classes = [
        "ssri",
        "snri",
        "tca",
        "statin",
        "anticoagulant",
        "opioid",
        "nsaid",
        "beta_blocker",
        "ace_inhibitor",
        "ppi",
    ]
    conditions_pool = [
        "diabetes",
        "heart_disease",
        "liver_disease",
        "kidney_disease",
        "hypertension",
    ]
    cyp2d6_alleles = ["*1", "*2", "*4", "*10", "*17", "*41"]
    cyp2c19_alleles = ["*1", "*2", "*17"]
    metabolizer_types = ["poor", "intermediate", "normal", "rapid", "ultrarapid"]

    age = int(rng.randint(18, 90))
    weight = float(rng.uniform(45, 130))
    height = float(rng.uniform(150, 200))
    bmi = round(weight / ((height / 100) ** 2), 1)

    num_conditions = int(rng.randint(0, 4))
    conditions = list(
        rng.choice(conditions_pool, size=min(num_conditions, len(conditions_pool)), replace=False)
    )

    return {
        "genetic_profile": {
            "CYP2D6": list(rng.choice(cyp2d6_alleles, size=int(rng.randint(1, 3)), replace=False)),
            "CYP2C19": list(
                rng.choice(cyp2c19_alleles, size=int(rng.randint(1, 2)), replace=False)
            ),
            "CYP3A4": ["*1"],
            "CYP2C9": ["*1"],
            "VKORC1": [],
            "DPYD": ["*1"],
            "TPMT": ["*1"],
            "UGT1A1": ["*1"],
            "SLCO1B1": ["*1A"],
            "HLA-B": [],
        },
        "metabolizer_phenotype": str(rng.choice(metabolizer_types)),
        "demographics": {
            "age": age,
            "weight_kg": round(weight, 1),
            "height_cm": round(height, 1),
            "bmi": bmi,
            "sex": str(rng.choice(["male", "female"])),
            "ethnicity": "unknown",
        },
        "drug": {
            "name": f"Drug_{rng.randint(1, 100)}",
            "drug_class": str(rng.choice(drug_classes)),
            "dosage_mg": float(rng.choice([10, 20, 25, 50, 100, 200])),
            "max_dosage_mg": 1000.0,
        },
        "medical_history": {
            "num_current_medications": int(rng.randint(0, 10)),
            "num_allergies": int(rng.randint(0, 5)),
            "num_adverse_reactions": int(rng.randint(0, 3)),
            "conditions": conditions,
            "pregnant": False,
            "age": age,
        },
    }


def run_benchmark(
    iterations: int = 1000,
    sla_threshold_ms: float = 100.0,
    warmup_iterations: int = 50,
) -> BenchmarkResult:
    """Run the latency benchmark.

    Measures model inference latency (feature processing + prediction).
    """
    # Import here to measure import time too
    from rx_predict.models.drug_response_model import DrugResponseModel

    print("Initializing model...")
    model = DrugResponseModel()

    init_start = time.perf_counter()
    model.build_default_model()
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"Model initialization: {init_time:.1f}ms")

    warmup_time = model.warm_up()
    print(f"Model warm-up avg: {warmup_time:.3f}ms")

    # Generate test data
    rng = np.random.RandomState(42)
    test_patients = [generate_sample_patient(rng) for _ in range(iterations)]

    # Warmup phase (not counted)
    print(f"\nWarmup phase ({warmup_iterations} iterations)...")
    for i in range(warmup_iterations):
        model.predict(test_patients[i % len(test_patients)])

    # Benchmark phase
    print(f"Benchmark phase ({iterations} iterations)...")
    latencies: list[float] = []

    total_start = time.perf_counter()
    for patient in test_patients:
        start = time.perf_counter()
        result = model.predict(patient)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
    total_elapsed = time.perf_counter() - total_start

    # Calculate statistics
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    mean = statistics.mean(latencies)
    std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    throughput = iterations / total_elapsed

    sla_pass = p99 <= sla_threshold_ms

    return BenchmarkResult(
        iterations=iterations,
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        min_ms=round(latencies[0], 3),
        max_ms=round(latencies[-1], 3),
        mean_ms=round(mean, 3),
        std_ms=round(std, 3),
        throughput_rps=round(throughput, 1),
        sla_pass=sla_pass,
        sla_threshold_ms=sla_threshold_ms,
    )


def run_batch_benchmark(
    batch_sizes: list[int] | None = None,
    iterations_per_size: int = 50,
) -> dict:
    """Benchmark batch prediction at various batch sizes."""
    from rx_predict.models.drug_response_model import DrugResponseModel

    if batch_sizes is None:
        batch_sizes = [1, 5, 10, 25, 50, 100]

    model = DrugResponseModel()
    model.build_default_model()
    model.warm_up()

    rng = np.random.RandomState(42)
    results = {}

    for batch_size in batch_sizes:
        latencies = []
        for _ in range(iterations_per_size):
            patients = [generate_sample_patient(rng) for _ in range(batch_size)]
            start = time.perf_counter()
            model.predict_batch(patients)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        latencies.sort()
        results[batch_size] = {
            "p50_ms": round(latencies[int(len(latencies) * 0.5)], 3),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
            "p99_ms": round(latencies[-1], 3),
            "mean_ms": round(statistics.mean(latencies), 3),
            "per_item_ms": round(statistics.mean(latencies) / batch_size, 3),
        }

    return results


def print_report(result: BenchmarkResult) -> None:
    """Print a formatted benchmark report."""
    print("\n" + "=" * 60)
    print("  RxPredict Latency Benchmark Report")
    print("=" * 60)
    print(f"  Iterations:     {result.iterations}")
    print(f"  Throughput:     {result.throughput_rps} req/s")
    print("-" * 60)
    print(f"  p50 latency:    {result.p50_ms:.3f} ms")
    print(f"  p95 latency:    {result.p95_ms:.3f} ms")
    print(f"  p99 latency:    {result.p99_ms:.3f} ms")
    print(f"  Min latency:    {result.min_ms:.3f} ms")
    print(f"  Max latency:    {result.max_ms:.3f} ms")
    print(f"  Mean latency:   {result.mean_ms:.3f} ms")
    print(f"  Std deviation:  {result.std_ms:.3f} ms")
    print("-" * 60)
    print(f"  SLA threshold:  {result.sla_threshold_ms:.0f} ms")
    sla_status = "PASS" if result.sla_pass else "FAIL"
    print(f"  SLA status:     {sla_status}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RxPredict latency benchmark")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--fail-threshold", type=float, default=100.0, help="p99 latency threshold in ms"
    )
    parser.add_argument("--batch", action="store_true", help="Also run batch benchmark")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    result = run_benchmark(
        iterations=args.iterations,
        sla_threshold_ms=args.fail_threshold,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "single_prediction": {
                        "iterations": result.iterations,
                        "p50_ms": result.p50_ms,
                        "p95_ms": result.p95_ms,
                        "p99_ms": result.p99_ms,
                        "mean_ms": result.mean_ms,
                        "throughput_rps": result.throughput_rps,
                        "sla_pass": result.sla_pass,
                    }
                },
                indent=2,
            )
        )
    else:
        print_report(result)

    if args.batch:
        print("\n  Batch Benchmark Results:")
        print("-" * 60)
        batch_results = run_batch_benchmark()
        for size, metrics in batch_results.items():
            print(
                f"  Batch size {size:>3}: p50={metrics['p50_ms']:.1f}ms  "
                f"p95={metrics['p95_ms']:.1f}ms  "
                f"per_item={metrics['per_item_ms']:.1f}ms"
            )
        print("=" * 60)

    if not result.sla_pass:
        print(f"\nERROR: SLA VIOLATION - p99 latency {result.p99_ms}ms > {args.fail_threshold}ms")
        sys.exit(1)
    else:
        print(f"\nSLA COMPLIANT - p99 latency {result.p99_ms}ms <= {args.fail_threshold}ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
