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
    parser.add_argument("--validator", default="on", choices=["on", "off"], help="Enable/disable validator")
    parser.add_argument("--threshold", type=float, default=0.85, help="Cache cosine similarity threshold")
    parser.add_argument("--output_dir", default="runs", help="Directory to save replay JSON files")
    parser.add_argument("--results_csv", default="runs/results.csv", help="Path to write summary CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_exists = os.path.exists(args.results_csv)

    fieldnames = [
        "seed", "mode", "robots", "steps", "cache", "validator", "threshold",
        "repair_success_rate", "mean_recovery_steps", "cost_delta_pct",
        "llm_calls", "llm_tokens", "cache_hits", "incidents_total"
    ]

    with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()

        for seed in args.seeds:
            run_id = f"{args.mode}_s{seed}_r{args.robots}"
            config = SimulationConfig(
                map_path=args.map,
                num_robots=args.robots,
                steps=args.steps,
                seed=seed,
                mode=args.mode,
                cache_enabled=(args.cache == "on"),
                validator_enabled=(args.validator == "on"),
                cache_threshold=args.threshold,
                run_id=run_id,
                output_dir=args.output_dir,
            )
            print(f"Running experiment: {run_id} (mode={args.mode}, seed={seed})...")
            replay = run(config)
            m = replay.metrics

            writer.writerow({
                "seed": seed,
                "mode": args.mode,
                "robots": args.robots,
                "steps": args.steps,
                "cache": args.cache,
                "validator": args.validator,
                "threshold": args.threshold,
                "repair_success_rate": m.repair_success_rate,
                "mean_recovery_steps": m.mean_recovery_steps,
                "cost_delta_pct": m.cost_delta_pct,
                "llm_calls": m.llm_calls,
                "llm_tokens": m.llm_tokens,
                "cache_hits": m.cache_hits,
                "incidents_total": len(replay.incidents),
            })
            f.flush()
            print(f"Finished {run_id}: Success={m.repair_success_rate*100:.1f}%, MeanRecovery={m.mean_recovery_steps} steps, CacheHits={m.cache_hits}")


if __name__ == "__main__":
    main()
