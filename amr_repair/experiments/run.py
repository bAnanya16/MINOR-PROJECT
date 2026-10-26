import os
import sys
import csv
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from amr_repair.loop.controller import SimulationConfig, run


def parse_args():
    parser = argparse.ArgumentParser(description="Lifelong MAPF Repair Experiment Runner")
    parser.add_argument("--map", default="maps/warehouse-10-20-10-2-1.map", help="Path to MovingAI .map")
    parser.add_argument("--robots", type=int, default=100, help="Number of robots")
    parser.add_argument("--steps", type=int, default=300, help="Simulation steps")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate")
    parser.add_argument("--mode", default="llm", choices=["llm", "baseline_pibt", "baseline_lns", "baseline_pbs"])
    parser.add_argument("--cache", default="on", choices=["on", "off"], help="Enable/disable strategy cache")
    parser.add_argument("--capacity", type=int, default=500, help="Cache capacity limit (LRU eviction)")
    parser.add_argument("--validator", default="on", choices=["on", "off"], help="Enable/disable validator gating")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cache cosine similarity threshold")
    parser.add_argument("--rate", type=float, default=0.05, help="Disruption injection rate per step")
    parser.add_argument("--overlap", default="same_aisle", choices=["none", "same_aisle", "same_robot"], help="Disruption overlap mode")
    parser.add_argument("--block_size", type=int, default=3, help="Max contiguous aisle blockage length")
    parser.add_argument("--output_dir", default="runs", help="Directory to save replay JSON files")
    parser.add_argument("--results_csv", default="runs/results.csv", help="Path to write summary CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_exists = os.path.exists(args.results_csv)

    fieldnames = [
        "seed", "mode", "robots", "steps", "overlap", "cache", "capacity", "validator", "threshold",
        "repair_success_rate", "mean_recovery_steps", "cost_delta_pct", "sum_of_costs_delta", "makespan_delta",
        "llm_calls", "llm_tokens", "llm_calls_avoided", "mean_llm_latency_ms",
        "cache_lookups", "cache_hits", "cache_misses", "cache_hit_rate_pct",
        "dry_run_successes", "dry_run_failures", "verified_cache_reuse_count", "verified_reuse_rate_pct",
        "cache_insertions", "cache_evictions",
        "mean_solver_latency_ms", "mean_validator_latency_ms", "mean_total_repair_latency_ms",
        "safety_violations_intercepted", "executed_safety_violations", "failed_repair_attempts", "fallback_invocations", "emergency_holds",
        "incidents_total", "incidents_isolated", "incidents_interacting"
    ]

    with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for seed in args.seeds:
            run_id = f"{args.mode}_s{seed}_r{args.robots}_{args.overlap}"
            config = SimulationConfig(
                map_path=args.map,
                num_robots=args.robots,
                steps=args.steps,
                seed=seed,
                mode=args.mode,
                disruption_rate=args.rate,
                overlap_mode=args.overlap,
                max_block_size=args.block_size,
                cache_enabled=(args.cache == "on"),
                cache_capacity=args.capacity,
                validator_enabled=(args.validator == "on"),
                cache_threshold=args.threshold,
                run_id=run_id,
                output_dir=args.output_dir,
            )
            print(f"Running experiment: {run_id} (mode={args.mode}, seed={seed}, overlap={args.overlap})...")
            replay = run(config)
            m = replay.metrics

            writer.writerow({
                "seed": seed,
                "mode": args.mode,
                "robots": args.robots,
                "steps": args.steps,
                "overlap": args.overlap,
                "cache": args.cache,
                "capacity": args.capacity,
                "validator": args.validator,
                "threshold": args.threshold,
                "repair_success_rate": m.repair_success_rate,
                "mean_recovery_steps": m.mean_recovery_steps,
                "cost_delta_pct": m.cost_delta_pct,
                "sum_of_costs_delta": m.sum_of_costs_delta,
                "makespan_delta": m.makespan_delta,
                "llm_calls": m.llm_calls,
                "llm_tokens": m.llm_tokens,
                "llm_calls_avoided": m.llm_calls_avoided,
                "mean_llm_latency_ms": m.mean_llm_latency_ms,
                "cache_lookups": m.cache_lookups,
                "cache_hits": m.cache_hits,
                "cache_misses": m.cache_misses,
                "cache_hit_rate_pct": m.cache_hit_rate_pct,
                "dry_run_successes": m.dry_run_successes,
                "dry_run_failures": m.dry_run_failures,
                "verified_cache_reuse_count": m.verified_cache_reuse_count,
                "verified_reuse_rate_pct": m.verified_reuse_rate_pct,
                "cache_insertions": m.cache_insertions,
                "cache_evictions": m.cache_evictions,
                "mean_solver_latency_ms": m.mean_solver_latency_ms,
                "mean_validator_latency_ms": m.mean_validator_latency_ms,
                "mean_total_repair_latency_ms": m.mean_total_repair_latency_ms,
                "safety_violations_intercepted": m.safety_violations_intercepted,
                "executed_safety_violations": m.executed_safety_violations,
                "failed_repair_attempts": m.failed_repair_attempts,
                "fallback_invocations": m.fallback_invocations,
                "emergency_holds": m.emergency_holds,
                "incidents_total": m.incidents_total,
                "incidents_isolated": m.incidents_isolated,
                "incidents_interacting": m.incidents_interacting,
            })
            f.flush()
            print(
                f"Finished {run_id}: Incidents={m.incidents_total} (Iso={m.incidents_isolated}, Int={m.incidents_interacting}), "
                f"Success={m.repair_success_rate*100:.1f}%, MeanRecovery={m.recovery_steps if hasattr(m, 'recovery_steps') else m.mean_recovery_steps} steps, "
                f"CacheHits={m.cache_hits}/{m.cache_lookups} ({m.cache_hit_rate_pct}%), InterceptedViolations={m.safety_violations_intercepted}"
            )


if __name__ == "__main__":
    main()
